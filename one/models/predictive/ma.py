import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import zscore

from one.models.base import Model



class MovingAverageModel(Model):
    support_multivariate = False
    def __init__(self, window: int = 10):
        self.window = window

    def fit(self, *args, **kwargs):
        return

    def get_scores(self, data, normalize=False):
        # if multivariate
        if data.ndim > 1 and data.shape[1] > 1:
            return self._handle_multivariate(data, [self]*data.shape[1])

        if data.ndim > 1 and data.shape[1] == 1:
            data = data.flatten()

        E = np.zeros(len(data))
        s_window = sliding_window_view(data, self.window)[:-1]

        E[self.window:] = data[self.window:] - np.mean(s_window, axis=1)
        
        if normalize: E = zscore(E)

        return np.abs(E)

