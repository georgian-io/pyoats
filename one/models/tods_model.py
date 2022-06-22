from typing import Any

import numpy as np
import numpy.typing as npt
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler
import optuna

from one.models.base import Model


class TODSModel(Model):
    def __init__(self, model_cls, window: int = 10, n_steps: int = 1):
        self.window = window
        self.n_steps = n_steps

        self.model_cls = model_cls
        self.model = model_cls(window_size=self.window, step_size=self.n_steps)

    @property
    def model_name(self):
        return type(self).__name__

    def __repr__(self):
        r = {}
        r.update({"model_name": self.model_name})
        r.update({"window": self.window})
        r.update({"n_steps": self.n_steps})

        return str(r)

    def fit(self, train_data: npt.NDArray[Any]):
        if train_data.ndim == 1:
            train_data = np.reshape(train_data, (-1, 1))

        self.model.fit(train_data)

    def get_scores(self, test_data: npt.NDArray[Any]):
        if test_data.ndim == 1:
            test_data = np.reshape(test_data, (-1, 1))

        # TODO: Either Fix this or the Experiment script!!
        return self.model.predict_score(test_data)

    def get_classification(self, test_data: npt.NDArray[Any]):
        if test_data.ndim == 1:
            test_data = np.reshape(test_data, (-1, 1))
        return self.model.predict(test_data)

    def hyperopt_ws(self, *args, **kwargs):
        # TODO: think about metric we use as target
        pass

    def hyperopt_model(self, *args, **kwargs):
        pass
