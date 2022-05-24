from typing import Any, Tuple

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler

from one.models.base import Model



class SimpleDartsModel(Model):
    def __init__(self,
                 model,
                 window: int,
                 n_steps: int):

        self.window = window
        self.n_steps = n_steps

        self.model = model
        self.transformer = Scaler()


    def fit(self, train_data: npt.NDArray[Any]):
        tr = self._scale_series(train_data)

        self.model.fit(TimeSeries.from_values(tr))


    def get_scores(self, test_data: npt.NDArray[Any]) -> Tuple[npt.NDArray[Any]]:
        # TODO: if makes sense, have a base class for this and DartsModel...

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


    def get_classification(self):
        pass


    def _scale_series(self, series:npt.NDArray[Any]):
        series = TimeSeries.from_values(series)
        series = self.transformer.fit_transform(series)

        return series.pd_series().to_numpy().astype(np.float32)

