from typing import Any, Tuple
from functools import partial
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler
import optuna

from one.models.base import Model


class SimpleDartsModel(Model):
    def __init__(self, model_cls, window: int, n_steps: int, lags: int):
        self.window = window
        self.n_steps = n_steps
        self.lags = lags

        self.model_cls = model_cls
        self.model = model_cls(self.lags)
        self.transformer = Scaler()
        self.params = None

    @property
    def model_name(self):
        return type(self).__name__

    def __repr__(self):
        r = {}
        r.update({"model_name": self.model_name})
        r.update({"window": self.window})
        r.update({"n_steps": self.n_steps})
        r.update({"lags": self.lags})
        r.update({"model_params": {} if not self.params else self.params})

        return str(r)

    def hyperopt_model(
        self,
        train_data: npt.NDArray[Any],
        test_data: npt.NDArray[Any],
        n_trials: int = 30,
        n_jobs: int = -1
    ):
        # TODO: we can probably merge this with the hyperparam tuning method for window size

        # obj
        obj = partial(
            self._model_objective,
            train_data=train_data,
            test_data=test_data,
        )

        study = optuna.create_study()
        study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs)

        self.params = study.best_params

        self.model = self.model_cls(self.lags, **self.params)

    def hyperopt_ws(
        self,
        train_data: npt.NDArray[any],
        test_data: npt.NDArray[any],
        n_trials: int = 30,
        n_jobs: int = -1
    ):
        obj = partial(
            self._ws_objective,
            train_data=train_data,
            test_data=test_data,
        )

        study = optuna.create_study()
        study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs)

        w = study.best_params.get("w")
        s = study.best_params.get("s")
        l = study.best_params.get("l")

        self.window = w
        self.n_steps = s
        self.lags = l

        self.model = self.model_cls(l)

    def _ws_objective(
        self,
        trial,
        train_data: npt.NDArray[any],
        test_data: npt.NDArray[any],
    ):
        w_high = int(0.25 * len(train_data))

        window = trial.suggest_int("w", 20, w_high, 5)
        n_steps = trial.suggest_int("s", 1, 20)
        lags = trial.suggest_int("l", 1, 20 - 1)

        cls = self.__class__(window, n_steps, lags)
        cls.model = cls.model_cls(lags)
        cls.fit(train_data)
        _, res, _ = cls.get_scores(test_data)

        return np.sum(res**2)

    def fit(self, train_data: npt.NDArray[Any]):
        if self.model is None:
            print("Unspecified hyperparameters, please run hyperopt_ws()")
            return

        tr = self._scale_series(train_data)

        self.model.fit(TimeSeries.from_values(tr))

    def get_scores(self, test_data: npt.NDArray[Any]) -> Tuple[npt.NDArray[Any]]:
        # TODO: if makes sense, have a base class for this and DartsModel...

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
        anom = np.absolute(residual)

        i_preds = TimeSeries.from_values(preds[: len(tdata_trim)])
        i_preds = self.transformer.inverse_transform(i_preds).pd_series().to_numpy()

        return anom, residual, i_preds

    def get_classification(self):
        pass

    def _scale_series(self, series: npt.NDArray[Any]):
        series = TimeSeries.from_values(series)
        series = self.transformer.fit_transform(series)

        return series.pd_series().to_numpy().astype(np.float32)
