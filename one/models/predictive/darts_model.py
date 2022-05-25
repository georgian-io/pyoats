from typing import Any, Tuple
from functools import partial

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler
from torch.cuda import device_count
import optuna

from one.models.base import Model
from one.utils import get_default_early_stopping


class DartsModel(Model):
    def __init__(
        self,
        model_cls,
        window: int,
        n_steps: int,
        use_gpu: bool,
        val_split: float = 0.05,
        rnn_model: str = None,
    ):

        self.window = window
        self.n_steps = n_steps
        self.use_gpu = use_gpu
        self.val_split = val_split
        self.val_split_mem = val_split
        self.rnn_model = rnn_model

        self.model_cls = model_cls

        # TODO: maybe seperate these into another class? fine for now...
        self.transformer = Scaler()

        self._init_model()

    def _init_model(self, **kwargs):
        if self.rnn_model:
            self.model = self.model_cls(
                self.window,
                training_length=self.window,
                pl_trainer_kwargs=self._get_trainer_kwargs(),
                model=self.rnn_model,
                **kwargs
            )
        else:
            self.model = self.model_cls(
                self.window,
                self.n_steps,
                pl_trainer_kwargs=self._get_trainer_kwargs(),
                **kwargs
            )

    def _get_trainer_kwargs(self) -> dict:
        d = {}

        if self.use_gpu:
            d.update({"accelerator": "gpu", "gpus": [i for i in range(device_count())]})

        d.update({"callbacks": [get_default_early_stopping()]})

        return d

    def hyperopt_ws(
        self,
        train_data: npt.NDArray[any],
        test_data: npt.NDArray[any],
        n_trials: int = 30,
    ):
        obj = partial(
            self._ws_objective,
            train_data=train_data,
            test_data=test_data,
            model_cls=self.model_cls,
        )

        study = optuna.create_study()
        study.optimize(obj, n_trials=n_trials)

        w = study.best_params.get("w")
        s = study.best_params.get("s")

        self.val_split = max(self.val_split, (w + s) / len(train_data) + 0.01)

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
        test_data: npt.NDArray[any],
        model_cls,
    ):
        w_high = int(0.25 * len(train_data))
        self.window = trial.suggest_int("w", 20, w_high, 5)
        self.n_steps = trial.suggest_int("s", 1, 20)

        self.val_split = max(
            self.val_split_mem, (self.window + self.n_steps) / len(train_data) + 0.01
        )

        self._init_model()
        self.fit(train_data)
        _, res, _ = self.get_scores(test_data)

        return np.sum(res**2)

    def fit(self, train_data: npt.NDArray[Any]):
        train_data = self._scale_series(train_data)
        tr, val = self._get_train_val_split(train_data, self.val_split)

        self.model.fit(
            TimeSeries.from_values(tr),
            val_series=TimeSeries.from_values(val),
            epochs=100,
            num_loader_workers=4,
        )

    def get_scores(self, test_data: npt.NDArray[Any]) -> Tuple[npt.NDArray[Any]]:
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

    def _get_train_val_split(
        self, series: npt.NDArray[Any], pct_val: float
    ) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        arr_len = len(series)
        split_at = int(arr_len * (1 - pct_val))

        return series[:split_at], series[split_at:]

    def _scale_series(self, series: npt.NDArray[Any]):
        series = TimeSeries.from_values(series)
        series = self.transformer.fit_transform(series)

        return series.pd_series().to_numpy().astype(np.float32)
