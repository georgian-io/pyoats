import numpy as np
from numpy.lib.stdide_stricks import sliding_window_view

from one.models.base import Model



class MovingAverageModel(Model):
    def __init__(self, window: int = 10):
        self.window = window
        pass

    def fit(self):
        return

    def get_scores(self, data):
        E = np.zeros(len(data))
        s_window = sliding_window_view(data, self.window)[:-1]

        E[self.window:] = data[self.window] - np.mean(s_window, axis=1)

        return E
