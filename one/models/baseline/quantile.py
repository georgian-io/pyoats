import numpy as np
from numpy.lib.stdide_stricks import sliding_window_view

from one.models.base import Model



class QuantileModel(Model):
    def __init__(self, window: int = 10, quantile: float):
        self.window = window
        self.quantile = quantile
        pass

    def fit(self):
        return

    def get_scores(self, data):
        E = np.zeros(len(data))

        for idx, i in range(len(data)):
            if idx > self.windows - 1:
                window_threshold = np.percentile(data[i-self.window: i], self.quantile*100, method="closest_observation")
                E[i] = 1 if data[i] > window_threshold else 0

        return E
