from typing import Protocol

import numpy as np


class Preprocessor(Protocol):
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def transform(self, *args, **kwargs):
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