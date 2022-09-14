from typing import Protocol, Any

import numpy as np


class Threshold(Protocol):
    def fit(self):
        raise NotImplementedError

    def get_scores(self):
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.get_threshold(*args, **kwargs)

    def _handle_multivariate(self, data, thresholders, **kwargs):
        scores = []
        for idx, series in enumerate(data.T):
            scores.append(thresholders[idx].get_threshold(series, **kwargs))

        return np.array(scores).T

    def _pseudo_mv_fit(self, data, **kwargs):
        thresholders = []
        for _ in range(data.shape[1]):
            thresholders.append(self.__class__(**self.__dict__))

        for idx, series in enumerate(data.T):
            thresholders[idx].fit(series, **kwargs)

        return thresholders
