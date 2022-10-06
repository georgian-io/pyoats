from typing import Protocol, Any

import numpy as np


class Threshold(Protocol):
    """Base class for thresholders

    Preprocessor object must take a `fit()` method as well as `get_threshold()`.
    If no fitting is required, concrete classes can simply return when `fit()` is called.

    Shape of returned array must be:
        Univariate: (t, ) & (t, 1) --> (t, )
        Multivariate: (t, n) --> (t, n)
    """

    def fit(self):
        raise NotImplementedError

    def get_threshold(self):
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.get_threshold(*args, **kwargs)

    def _handle_multivariate(self, data, thresholders, **kwargs):
        scores = []
        for idx, series in enumerate(data.T):
            print(series.shape, len(series.shape))
            scores.append(thresholders[idx].get_threshold(series, **kwargs))

        return np.array(scores).T

    def _pseudo_mv_fit(self, data, **kwargs):
        thresholders = []
        for _ in range(data.shape[1]):
            params = self.__dict__
            thresholders.append(self.__class__(**params))

        for idx, series in enumerate(data.T):
            thresholders[idx].fit(series, **kwargs)

        return thresholders
