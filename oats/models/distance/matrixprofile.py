"""
Matrix Profile
-----------------
"""

from oats.models._base import Model
from stumpy import stump, scrump, mstump, gpu_stump
import numpy as np


class MatrixProfileModel(Model):
    def __init__(self, window: int = 10, use_gpu: bool = False, approx: bool = True):
        self.window = window
        self.use_gpu = use_gpu
        self.approx = approx

    def fit(self, *args, **kwargs):
        return

    def get_scores(self, data):
        multivar = True if data.ndim > 1 and data.shape[1] > 1 else False

        # univariate in 2-D matrix
        if data.ndim > 1 and data.shape[1] == 1:
            data = data.flatten()

        if multivar:
            model = mstump
            data = data.T
            get_scores = lambda arr: arr[0].T

            scores = model(data, self.window)
        elif self.use_gpu:
            model = gpu_stump
            get_scores = lambda arr: arr[:, 0]

            scores = model(data, self.window)
        else:
            if self.approx:
                model = scrump
                get_scores = lambda arr: arr

                scores = model(
                    data, self.window, percentage=0.01, pre_scrump=True, s=None
                )
                scores.update()
                scores = scores.P_

            else:
                model = stump
                get_scores = lambda arr: arr[:, 0]

                scores = model(data, self.window)

        scores = get_scores(scores)

        if multivar:
            scores = np.append(
                np.zeros((self.window - 1, data.T.shape[1])), scores, axis=0
            )
        else:
            scores = np.append(np.zeros(self.window - 1), scores)

        return scores
