from typing import Protocol
import numpy.typing as npt
import numpy as np


class Model(Protocol):
    """Base class for Models

    Preprocessors are any function that transforms timeseries T to T'.

    Model object must take a `fit()` method as well as `get_scores()`.
    If no fitting is required, concrete classes can simply return when `fit()` is called.

    Shape of returned array must be:
        Univariate: (t, ) & (t, 1) --> (t, )
        Multivariate: (t, n) --> (t, n)

    Validity of return shape can be tested by adding concrete class in `/test/test_scorer.py`

    Example:
        pproc = SpectralResidual()
        transformed_train = pproc.transform(train)

    """

    def fit(self):
        raise NotImplementedError

    def get_scores(self):
        raise NotImplementedError

    @property
    def _model_name(self):
        name = type(self).__name__
        if self.rnn_model:
            return f"{name}_{self.rnn_model}"

        return name

    def _handle_multivariate(self, data, models):
        scores = []
        for idx, series in enumerate(data.T):
            scores.append(models[idx].get_scores(series))

        return np.array(scores).T

    def _pseudo_mv_train(self, data):
        models = []
        for _ in range(data.shape[1]):
            params = self.__dict__
            if "model_cls" in params:
                params.pop("model_cls")
            models.append(self.__class__(**params))

        for idx, series in enumerate(data.T):
            models[idx].fit(series)

        return models
