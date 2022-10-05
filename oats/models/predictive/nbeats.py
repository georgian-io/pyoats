"""
N-BEATS
-----------------
"""
from typing import Any
from functools import partial

import numpy as np
import numpy.typing as npt
import optuna
from darts import models

from oats.models._darts_model import DartsModel


class NBEATSModel(DartsModel):
    """N-BEATS Model (Neural Basis Expansion Analysis Time Series Forecasting)

    Using N-BEATS as a predictor. Anomalies scores are deviations from predictions.

    Reference: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nbeats.html
    """

    def __init__(
        self,
        window: int = 10,
        n_steps: int = 1,
        use_gpu: bool = False,
        val_split: float = 0.2,
        **kwargs
    ):
        """
        initialization also accepts any parameters used by: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nbeats.html

        Args:
            window (int, optional): rolling window size to feed into the predictor. Defaults to 10.
            n_steps (int, optional): number of steps to predict forward. Defaults to 1.
            use_gpu (bool, optional): whether to use GPU. Defaults to False.
            val_split (float, optional): proportion of data points reserved for validation. Defaults to 0.2.
        """

        model = models.NBEATSModel

        super().__init__(model, window, n_steps, use_gpu, val_split, **kwargs)

    def _model_objective(self, trial, train_data: npt.NDArray[Any]):
        params = {
            "num_blocks": trial.suggest_int("num_blocks", 1, 2),
            "num_stacks": trial.suggest_int("num_stacks", 2, 32),
            "num_layers": trial.suggest_int("num_layers", 1, 16),
            "layer_widths": trial.suggest_int("layer_widths", 128, 512),
            "expansion_coefficient_dim": trial.suggest_int(
                "expansion_coefficient_dim", 1, 10
            ),
            "batch_size": trial.suggest_int(
                "batch_size", 1, (len(train_data) - self.window) // self.n_steps // 4
            ),
        }

        return self._get_hyperopt_res(params, train_data)
