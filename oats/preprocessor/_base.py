from typing import Protocol

import numpy as np


class Preprocessor(Protocol):
    """Base class for Preprocessors.

    Preprocessors are any function that transforms timeseries T to T'.

    Preprocessor object must take a `fit()` method as well as `transform()`.
    If no fitting is required, concrete classes can simply return when `fit()` is called.

    Shape of returned array must be:
        Univariate: (t, ) & (t, 1) --> (t, )
        Multivariate: (t, n) --> (t, n)

    Validity of return shape can be tested by adding concrete class in `/test/test_preprocessor.py`

    Example:
        pproc = SpectralResidual()
        transformed_train = pproc.transform(train)
    """

    def fit(self, data, *args, **kwargs):
        raise NotImplementedError

    def transform(self, data, *args, **kwargs):
        raise NotImplementedError

    def _handle_multivariate(self, data, processors):
        scores = []
        for idx, series in enumerate(data.T):
            scores.append(processors[idx].transform(series))

        return np.array(scores).T

    def _pseudo_mv_fit(self, data):
        processors = []
        for _ in range(data.shape[1]):
            processors.append(self.__class__(**self.__dict__))

        for idx, series in enumerate(data.T):
            processors[idx].fit(series)

        return processors
