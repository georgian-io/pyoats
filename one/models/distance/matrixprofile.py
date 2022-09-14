from one.models.base import Model
from stumpy import stump, mstump, gpu_stump
import numpy as np


class MatrixProfileModel(Model):
    def __init__(self, window: int = 10, use_gpu: bool = False):
        self.window = window
        self.use_gpu = use_gpu

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
        elif self.use_gpu:
            model = gpu_stump
            get_scores = lambda arr: arr[:, 0]
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
