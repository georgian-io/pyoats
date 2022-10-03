"""
Implementation from: https://github.com/yzhao062/pyod
"""
from typing import Any

import numpy as np
import numpy.typing as npt
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler
from scipy.stats import zscore
from numpy.lib.stride_tricks import sliding_window_view

from oats.models._base import Model


class PyODModel(Model):
    def __init__(self, model_cls, window: int = 10, **kwargs):
        self.window = window
        self.params = kwargs

        self.model_cls = model_cls
        self.model = model_cls(**self.params)
        self.scaler = None

    @property
    def _model_name(self):
        return type(self).__name__

    def __repr__(self):
        r = {}
        r.update({"model_name": self._model_name})
        r.update({"window": self.window})

        return str(r)

    def fit(self, train_data: npt.NDArray[Any], **kwargs):
        if "epochs" in kwargs and hasattr(self, "IS_DL") and self.IS_DL:
            self.params["epochs"] = kwargs["epochs"]
            self.model = self.model_cls(**self.params)

        windows = self._get_window(train_data)
        self.model.fit(windows)

    def get_scores(self, test_data: npt.NDArray[Any], normalize=False):
        # Multivar
        multivar = True if test_data.ndim > 1 and test_data.shape[1] > 1 else False

        windows = self._get_window(test_data)
        scores = self.model.decision_function(windows)

        if normalize:
            scores = zscore(scores)
        scores = np.abs(scores)

        scores = np.append(np.zeros(self.window - 1), scores)
        if multivar:
            scores = np.tile(scores, (test_data.shape[1], 1)).T

        return scores

    def _get_window(self, data):
        # Univariate in 2-D
        data = self._scale_series(data)
        if data.ndim > 1 and data.shape[1] == 1:
            data = data.flatten()

        # Multivar
        multivar = True if data.ndim > 1 and data.shape[1] > 1 else False

        windows = sliding_window_view(data, self.window, axis=0)

        if multivar:
            n_samples, feats, n_t = windows.shape
            windows = windows.reshape(n_samples, n_t * feats)
        return windows

    def _scale_series(self, series):
        series = TimeSeries.from_values(series)

        if self.scaler is None:
            self.scaler = Scaler()
            self.scaler.fit(series)

        series = self.scaler.transform(series)

        return series.pd_dataframe().to_numpy().astype(np.float32)
