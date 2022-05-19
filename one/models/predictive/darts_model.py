from typing import Any, Tuple

from darts.datasets import TimeSeries
from darts.dataprocessing.transformers import Scaler
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view
from torch.cuda import device_count

from one.models.base import Model
from one.utils import get_default_early_stopping

class DartsModel(Model):
    def __init__(self,
                 model,
                 window: int,
                 n_steps: int,
                 use_gpu: bool,
                 val_split: float = 0.05):

        self.window = window
        self.n_steps = n_steps
        self.use_gpu = use_gpu
        self.val_split = val_split

        self.model = model
        self.transformer = Scaler()

        return

    def _get_trainer_kwargs(self) -> dict:
        d = {}

        if self.use_gpu:
            d.update(
                {
                    "accelerator": "gpu",
                    "gpus": [i for i in range(device_count())]
                }
            )

        d.update({"callbacks": get_default_early_stopping()})

        return d


    def fit(self, train_data: npt.NDArray[Any]):
        train_data = self._scale_series(train_data)
        tr, val = self._get_train_val_split(train_data, self.val_split)

        self.model.fit(TimeSeries.from_values(tr),
                       val_series = TimeSeries.from_values(val),
                       epochs = 100,
                       num_loader_workers = 4
                       )

    def get_scores(self, test_data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        test_data = self._scale_series(test_data)

        windows = sliding_window_view(test_data, self.window)
        windows = windws[::self.n_steps]

        preds = np.array([])

        seq = []
        for arr in enumerate(windows.astype(np.float32)):
            ts = TimeSeries.from_values(arr)
            seq.append(ts)

        scores = model.predict(n=self.n_steps, series=seq)

        for step in scores:
            preds = np.append(preds, step.pd_series().to_numpy())

        return preds

    def _get_train_val_split(self, series:npt.NDArray[Any], pct_val: float) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        arr_len = len(series)
        split_at = int(arr_len * (1-pct_val))

        return series[:split_at], series[split_at:]

    def _scale_series(self, series:npt.NDArray[Any]):
        series = TimeSeries.from_values(train_data)
        series = self.transformer.fit_transform(series)

        return series.pd_series().to_numpy().astype(np.float32)


