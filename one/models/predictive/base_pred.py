from typing import Protocol


class PredictiveModel(Protocol):
    def fit(self):
        raise NotImplementedError

    def get_scores(self):
        raise NotImplementedError

    def get_classification(self):
        raise NotImplementedError

    def set_contamination(self):
        raise NotImplementedError
