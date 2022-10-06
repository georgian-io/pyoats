"""
Implementation from: https://github.com/unit8co/darts
"""
from typing import Any, Tuple
from functools import partial

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler
from torch.cuda import device_count
from scipy.stats import zscore
import optuna

from oats.models._base import Model
from oats._utils.utils import get_default_early_stopping


class DartsModel(Model):
    _support_multivariate = True

    def __init__(
        self,
        model_cls,
        window: int,
        n_steps: int,
        use_gpu: bool,
        val_split: float = 0.2,
        rnn_model: str = None,
        **kwargs,
    ):

        self.window = window
        self.n_steps = n_steps
        self.use_gpu = use_gpu
        self.val_split = val_split
        self.val_split_mem = val_split
        self.rnn_model = rnn_model

        self.model_cls = model_cls

        # TODO: maybe seperate these into another class? fine for now...
        self.transformer = None

        self.params = None

        self._init_model(**kwargs)

    @property
    def _model_name(self):
        name = type(self).__name__
        if self.rnn_model:
            return f"{name}_{self.rnn_model}"

        return name

    def __repr__(self):
        r = {}
        r.update({"model_name": self._model_name})
        r.update({"window": self.window})
        r.update({"n_steps": self.n_steps})
        r.update({"val_split": self.val_split})
        r.update({"model_params": {} if not self.params else self.params})

        return str(r)

    def _init_model(self, **kwargs):
        if self.rnn_model:
            self.model = self.model_cls(
                self.window,
                training_length=self.window,
                pl_trainer_kwargs=self._get_trainer_kwargs(),
                model=self.rnn_model,
                **kwargs,
            )
            return

        self.model = self.model_cls(
            self.window,
            self.n_steps,
            pl_trainer_kwargs=self._get_trainer_kwargs(),
            **kwargs,
        )

    def _get_trainer_kwargs(self) -> dict:
        d = {}

        if self.use_gpu:
            d.update({"accelerator": "gpu", "gpus": [i for i in range(device_count())]})

        d.update({"callbacks": [get_default_early_stopping()]})
        return d

    def hyperopt_model(
        self,
        train_data: npt.NDArray[Any],
        n_trials: int = 30,
        n_jobs: int = 1,
    ):
        # TODO: we can probably merge this with the hyperparam tuning method for window size

        obj = partial(
            self._model_objective,
            train_data=train_data,
        )

        study = optuna.create_study()
        study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs)

        self.params = study.best_params
        self._init_model(**self.params)

    def hyperopt_ws(
        self,
        train_data: npt.NDArray[any],
        n_trials: int = 30,
        n_jobs: int = 1,
    ):
        obj = partial(
            self._ws_objective,
            train_data=train_data,
        )

        study = optuna.create_study()
        study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs)

        w = study.best_params.get("w")
        s = study.best_params.get("s")

        self.window = w
        self.n_steps = s
        self.val_split = max(
            self.val_split_mem, (self.window + self.n_steps) / len(train_data) + 0.01
        )

        self._init_model()

    def _ws_objective(
        self,
        trial,
        train_data: npt.NDArray[any],
    ):
        w_high = min(
            int(0.25 * len(train_data)), int(len(train_data) * self.val_split * 0.5)
        )

        window = trial.suggest_int("w", 20, w_high, 5)
        n_steps = trial.suggest_int("s", 1, 20)

        val_split = max(self.val_split_mem, (window + n_steps) / len(train_data) + 0.01)

        try:
            m = self.__class__(window, n_steps, self.use_gpu, val_split)
            m.fit(train_data)
        except RuntimeError:
            return 1e4

        _, val = self._get_train_val_split(train_data, self.val_split)

        try:
            _, res, _ = m.get_scores(val)
        except ValueError:
            return 1e4

        return np.sum(res**2)

    def fit(self, train_data: npt.NDArray[Any], epochs: int = 15, **kwargs):

        train_data = self._scale_series(train_data)
        tr, val = self._get_train_val_split(train_data, self.val_split)

        self.model.fit(
            TimeSeries.from_values(tr),
            val_series=TimeSeries.from_values(val),
            epochs=epochs,
            num_loader_workers=1,
            **kwargs,
        )

    def get_scores(
        self, test_data: npt.NDArray[np.float32], normalize=False, **kwargs
    ) -> Tuple[npt.NDArray[np.float32]]:
        test_data = self._scale_series(test_data)

        windows = sliding_window_view(test_data, self.window, axis=0)
        windows = windows[:: self.n_steps]
        multivar = True if test_data.ndim > 1 and test_data.shape[1] > 1 else False

        if test_data.ndim > 1 and test_data.shape[1] == 1:
            test_data = test_data.flatten()

        preds = np.ndarray(windows.shape[1]) if multivar else np.array([])

        seq = []
        for arr in windows.astype(np.float32):
            if multivar:
                arr = arr.T
            else:
                arr = arr.flatten()

            ts = TimeSeries.from_values(arr)
            seq.append(ts)

        scores = self.model.predict(n=self.n_steps, series=seq, **kwargs)

        for step in scores:
            if multivar:
                preds = np.vstack((preds, step.pd_dataframe().to_numpy()))
            else:
                preds = np.append(preds, step.pd_dataframe().to_numpy())

        tdata_trim = test_data[self.window :]

        preds = preds[: len(tdata_trim)]

        residual = preds - tdata_trim

        if normalize:
            residual = zscore(residual)

        residual = np.abs(residual)

        if multivar:
            residual = np.append(
                np.zeros((self.window, test_data.shape[1])), residual, axis=0
            )
        else:
            residual = np.append(np.zeros(self.window), residual)

        anom = np.absolute(residual)

        i_preds = TimeSeries.from_values(preds[: len(tdata_trim)])
        i_preds = self.transformer.inverse_transform(i_preds).pd_dataframe().to_numpy()

        self._preds = i_preds
        self._resudual = residual

        return anom

    def _get_train_val_split(
        self, series: npt.NDArray[Any], pct_val: float
    ) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        arr_len = len(series)
        split_at = int(arr_len * (1 - pct_val))

        return series[:split_at], series[split_at - self.window :]

    def _scale_series(self, series: npt.NDArray[Any]):
        series = TimeSeries.from_values(series)

        if self.transformer is None:
            self.transformer = Scaler()
            self.transformer.fit(series)

        series = self.transformer.transform(series)

        return series.pd_dataframe().to_numpy().astype(np.float32)

    def _get_hyperopt_res(self, params: dict, train_data):
        try:
            m = self.__class__(self.window, self.n_steps, self.use_gpu, self.val_split)
            m._init_model(**params)
            m.fit(train_data)

        except RuntimeError:
            return 1e4

        _, val = self._get_train_val_split(train_data, self.val_split)
        _, res, _ = m.get_scores(val)
        return np.sum(res**2)
