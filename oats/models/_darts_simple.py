"""
Implementation from: https://github.com/unit8co/darts
"""

from typing import Any, Tuple
from functools import partial
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler
from scipy.stats import zscore
import optuna

from oats.models._base import Model


class SimpleDartsModel(Model):
    def __init__(
        self, model_cls, window: int, n_steps: int, lags: int, val_split=0.2, **kwargs
    ):
        self.window = window
        self.n_steps = n_steps
        self.lags = lags
        self.val_split = val_split
        self.val_split_mem = val_split

        self.model_cls = model_cls
        self.model = model_cls(self.lags)
        self.transformer = None
        self.params = kwargs

    @property
    def _model_name(self):
        return type(self).__name__

    def __repr__(self):
        r = {}
        r.update({"model_name": self._model_name})
        r.update({"window": self.window})
        r.update({"n_steps": self.n_steps})
        r.update({"lags": self.lags})
        r.update({"model_params": {} if not self.params else self.params})

        return str(r)

    def hyperopt_model(
        self,
        train_data: npt.NDArray[Any],
        n_trials: int = 30,
        n_jobs: int = -1,
    ):
        # TODO: we can probably merge this with the hyperparam tuning method for window size

        # obj
        obj = partial(
            self._model_objective,
            train_data=train_data,
        )

        study = optuna.create_study()
        study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs)

        self.params = study.best_params

        self.model = self.model_cls(self.lags, **self.params)

    def hyperopt_ws(
        self,
        train_data: npt.NDArray[any],
        n_trials: int = 30,
        n_jobs: int = -1,
    ):
        obj = partial(
            self._ws_objective,
            train_data=train_data,
        )

        study = optuna.create_study()
        study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs)

        w = study.best_params.get("w")
        s = study.best_params.get("s")
        l = study.best_params.get("l")

        self.window = w
        self.n_steps = s
        self.lags = l

        self.val_split = max(
            self.val_split_mem, (self.window + self.n_steps) / len(train_data) + 0.01
        )

        self.model = self.model_cls(l)

    def _ws_objective(
        self,
        trial,
        train_data: npt.NDArray[any],
    ):
        w_high = max(
            int(0.25 * len(train_data)), int(len(train_data) * self.val_split * 0.5)
        )

        window = trial.suggest_int("w", 20, w_high, 5)
        n_steps = trial.suggest_int("s", 1, 20)
        lags = trial.suggest_int("l", 1, 20 - 1)

        val_split = min(
            self.val_split_mem, (self.window + self.n_steps) / len(train_data) + 0.01
        )

        cls = self.__class__(window, n_steps, lags, val_split)
        cls.model = cls.model_cls(lags)
        cls.fit(train_data)

        tr, val = cls._get_train_val_split(train_data, val_split)

        try:
            _, res, _ = cls.get_scores(val)
        except ValueError:
            return 1e4

        return np.sum(res**2)

    def fit(self, train_data: npt.NDArray[Any], *args, **kwargs):
        if self.model is None:
            print("Unspecified hyperparameters, please run hyperopt_ws()")
            return

        if train_data.ndim > 1 and train_data.shape[1] > 1:
            self._models = self._pseudo_mv_train(train_data)
            return

        train_data = self._scale_series(train_data)

        tr, val = self._get_train_val_split(train_data, self.val_split)

        self.model.fit(TimeSeries.from_values(tr))

    def get_scores(self, test_data: npt.NDArray[Any]) -> Tuple[npt.NDArray[Any]]:
        # TODO: if makes sense, have a base class for this and DartsModel...

        if test_data.ndim > 1 and test_data.shape[1] > 1:
            return self._handle_multivariate(test_data, self._models)

        test_data = self._scale_series(test_data)

        windows = sliding_window_view(test_data, self.window)
        windows = windows[:: self.n_steps]

        preds = np.array([])

        seq = []
        for arr in windows.astype(np.float32):
            ts = TimeSeries.from_values(arr)
            seq.append(ts)

        scores = self.model.predict(n=self.n_steps, series=seq)

        for step in scores:
            preds = np.append(preds, step.pd_series().to_numpy())

        tdata_trim = test_data[self.window :]

        preds = preds[: len(tdata_trim)]

        residual = preds - tdata_trim
        residual = np.abs(zscore(residual))
        residual = np.append(np.zeros(self.window), residual)

        anom = np.absolute(residual)

        i_preds = TimeSeries.from_values(preds)
        i_preds = self.transformer.inverse_transform(i_preds).pd_series().to_numpy()
        i_preds = np.append(np.zeros(self.window), i_preds)

        self._preds = i_preds
        self._residual = residual

        return anom

    def _scale_series(self, series: npt.NDArray[Any]):
        series = TimeSeries.from_values(series)

        if self.transformer is None:
            self.transformer = Scaler()
            self.transformer.fit(series)

        series = self.transformer.transform(series)

        return series.pd_series().to_numpy().astype(np.float32)

    def _get_train_val_split(
        self, series: npt.NDArray[Any], pct_val: float
    ) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        arr_len = len(series)
        split_at = int(arr_len * (1 - pct_val))

        return series[:split_at], series[split_at - self.window :]

    def _get_hyperopt_res(self, params: dict, train_data):
        try:
            m = self.__class__(self.window, self.n_steps, self.lags, self.val_split)
            m.model = m.model_cls(self.lags, **params)

            m.fit(train_data)

        except RuntimeError:
            return 1e4

        _, val = self._get_train_val_split(train_data, self.val_split)
        _, res, _ = m.get_scores(val)
        return np.sum(res**2)
