from typing import Protocol
import numpy.typing as npt


class Model(Protocol):
    def fit(self):
        raise NotImplementedError

    def get_scores(self):
        raise NotImplementedError

    def hyperopt_ws(self):
        raise NotImplementedError

    def hyperopt_model(self):
        raise NotImplementedError

    @property
    def model_name(self):
        name = type(self).__name__
        if self.rnn_model:
            return f"{name}_{self.rnn_model}"

        return name
