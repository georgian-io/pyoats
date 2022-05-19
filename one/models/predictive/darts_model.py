from typing import Any, Tuple

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler
from torch.cuda import device_count

from one.models.base import Model
from one.utils import get_default_early_stopping

class DartsModel(Model):
    def __init__(self,
                 model,
                 window: int,
                 n_steps: int,
                 val_split: float = 0.05):

        self.window = window
        self.n_steps = n_steps
        self.val_split = val_split

        self.model = model
        self.transformer = Scaler()

        return

    def _get_trainer_kwargs(self, use_gpu: bool) -> dict:
        d = {}

        if use_gpu:
            d.update(
                {
                    "accelerator": "gpu",
                    "gpus": [i for i in range(device_count())]
                }
            )

        d.update({"callbacks": [get_default_early_stopping()]})

        return d


    def fit(self, train_data: npt.NDArray[Any]):
        train_data = self._scale_series(train_data)
        tr, val = self._get_train_val_split(train_data, self.val_split)

        self.model.fit(TimeSeries.from_values(tr),
                       val_series = TimeSeries.from_values(val),
                       epochs = 100,
                       num_loader_workers = 4
                       )

    def get_scores(self, test_data: npt.NDArray[Any]) -> Tuple[npt.NDArray[Any]]:
        test_data = self._scale_series(test_data)

        windows = sliding_window_view(test_data, self.window)
        windows = windows[::self.n_steps]

        preds = np.array([])

        seq = []
        for arr in windows.astype(np.float32):
            ts = TimeSeries.from_values(arr)
            seq.append(ts)

        scores = self.model.predict(n=self.n_steps, series=seq)

        for step in scores:
            preds = np.append(preds, step.pd_series().to_numpy())
       
        tdata_trim = test_data[self.window:]
        
        preds = preds[:len(tdata_trim)]
        residual = preds - tdata_trim
        anom = np.absolute(residual)

        i_preds = TimeSeries.from_values(preds[:len(tdata_trim)])
        i_preds = self.transformer.inverse_transform(i_preds).pd_series().to_numpy()

        return anom, residual, i_preds

    def _get_train_val_split(self, series:npt.NDArray[Any], pct_val: float) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        arr_len = len(series)
        split_at = int(arr_len * (1-pct_val))

        return series[:split_at], series[split_at:]

    def _scale_series(self, series:npt.NDArray[Any]):
        series = TimeSeries.from_values(series)
        series = self.transformer.fit_transform(series)

        return series.pd_series().to_numpy().astype(np.float32)


