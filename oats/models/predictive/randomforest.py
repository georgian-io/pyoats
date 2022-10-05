"""
Random Forest
-----------------
"""
from typing import Any
from functools import partial

from darts import models
import numpy as np
import numpy.typing as npt
import optuna


from oats.models._darts_simple import SimpleDartsModel


class RandomForestModel(SimpleDartsModel):
    """Random Forest Model

    Using random forest regression as a predictor. Anomalies scores are deviations from predictions.

    Reference: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.random_forest.html
    """

    def __init__(
        self,
        window: int = 10,
        n_steps: int = 1,
        lags: int = 1,
        val_split: float = 0.2,
        **kwargs
    ):
        """
        initialization also accepts any parameters used by: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.random_forest.html

        Args:
            window (int, optional): rolling window size to feed into the predictor. Defaults to 10.
            n_steps (int, optional): number of steps to predict forward. Defaults to 1.
            lags (int, optional): number of lags. Defaults to 1.
            val_split (float, optional): proportion of data points reserved for validation; only used if using auto-tuning (not tested). Defaults to 0.
        """

        model = models.RandomForest

        super().__init__(model, window, n_steps, lags, val_split, **kwargs)

    def _model_objective(self, trial, train_data: npt.NDArray[Any]):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 1000),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2"]
            ),
            "max_depth": trial.suggest_int("max_depth", 1, 5000),
            # "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 2e-2)
        }

        return self._get_hyperopt_res(params, train_data)
