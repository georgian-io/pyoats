from typing import Any
from functools import partial

from darts import models
import numpy as np
import numpy.typing as npt
import optuna


from one.models.predictive.darts_simple import SimpleDartsModel


class RandomForestModel(SimpleDartsModel):
    def __init__(self, window: int = 10, n_steps: int = 1, lags: int = 1):

        model = models.RandomForest

        super().__init__(model, window, n_steps, lags)

    def _model_objective(
        self, trial, train_data: npt.NDArray[Any], test_data: npt.NDArray[Any]
    ):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 1000),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2"]
            ),
            "max_depth": trial.suggest_int("max_depth", 1, 5000),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 5e-2),
        }

        self.model = self.model_cls(self.lags, **params)
        self.fit(train_data)
        _, res, _ = self.get_scores(test_data)

        return np.sum(res**2)