from typing import Any

import numpy.typing as npt
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler
import optuna

from one.models.base import Model


class TODSModel(Model):
    def __init__(self, model_cls, window: int, n_stes: int):
        self.window = window
        self.n_steps = window

        self.model_cls = model_cls
        self.model = model_cls({"window_size": self.window, "step_size": self.n_steps})

    def fit(self, train_data: npt.NDArray[Any]):
        self.model.fit()

    def get_scores(self, train_data: npt.NDarray[Any]):
        return self.model.predict_score(train_data)

    def get_classification(self, train_data: npt.NDarray[Any]):
        return self.model.predict(train_data)

    def hyperopt_ws(self):
        # TODO: think about metric we use as target
        raise NotImplementedError

    def hyperopt_model(self):
        raise NotImplementedError
