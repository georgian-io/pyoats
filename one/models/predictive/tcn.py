from typing import Any
from functools import partial

from darts import models
import numpy as np
import numpy.typing as npt
import optuna

from one.models.predictive.darts_model import DartsModel


class TCNModel(DartsModel):
    def __init__(
        self,
        window: int = 10,
        n_steps: int = 1,
        use_gpu: bool = 1,
        val_split: float = 0.05,
    ):

        model = models.TCNModel

        super().__init__(model, window, n_steps, use_gpu, val_split)

    def _model_objective(
        self, trial, train_data: npt.NDArray[Any], test_data: npt.NDArray[Any]
    ):
        params = {
            "kernel_size": trial.suggest_int(
                "kernel_size", 2, min(32, self.window - 1)
            ),
            "num_filters": trial.suggest_int("num_filters", 2, 8),
            "weight_norm": trial.suggest_categorical("weight_norm", [True, False]),
            "dilation_base": trial.suggest_int("dilation_base", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "batch_size": trial.suggest_int(
                "batch_size", 1, len(train_data) - self.window // self.n_steps // 4
            ),
        }

        self._init_model(**params)
        self.fit(train_data)
        _, res, _ = self.get_scores(test_data)

        return np.sum(res**2)
