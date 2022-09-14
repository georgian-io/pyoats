import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from one.models.base import Model


class QuantileModel(Model):
    support_multivariate = False

    def __init__(self, window: int = 10, quantile: float = 0.98):
        self.window = window
        self.quantile = quantile
        pass

    def fit(self, *args, **kwargs):
        return

    def get_scores(self, data):
        if data.ndim > 1 and data.shape[1] > 1:
            return self._handle_multivariate(data, [self] * data.shape[1])

        E = np.zeros(len(data))

        for i in range(len(data)):
            if i > self.window - 1:
                window_threshold = np.percentile(
                    data[i - self.window : i],
                    self.quantile * 100,
                    method="closest_observation",
                )
                E[i] = 1 if data[i] > window_threshold else 0

        return E
