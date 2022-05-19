from typing import Protocol
import numpy.typing as npt

class Model(Protocol):
    def fit(self):
        raise NotImplementedError

    def get_scores(self):
        raise NotImplementedError

    def get_classification(self):
        raise NotImplementedError
