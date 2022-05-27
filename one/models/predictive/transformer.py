from typing import Any
from functools import partial

from darts import models
import numpy as np
import numpy.typing as npt
import optuna


from one.models.predictive.darts_model import DartsModel


class TransformerModel(DartsModel):
    def __init__(
        self, window: int, n_steps: int, use_gpu: bool, val_split: float = 0.05
    ):

        model = models.TransformerModel
        super().__init__(model, window, n_steps, use_gpu, val_split)

    def _model_objective(
        self, trial, train_data: npt.NDArray[Any], test_data: npt.NDArray[Any]
    ):
        params = {
            "d_model": trial.suggest_int("d_model", 16, 512, 8),
            "nhead": trial.suggest_int("nhead", 2, 8, 2),
            "num_encoder_layers": trial.suggest_int("num_encoder_layers", 2, 32, 2),
            "num_decoder_layers": trial.suggest_int("num_decoder_layers", 2, 32, 2),
            "dim_feedforward": trial.suggest_int("dim_feedforward", 256, 2048, 8),
            "batch_size": trial.suggest_int(
                "batch_size", 1, (len(train_data) - self.window) // self.n_steps // 4
            ),
        }

        return self._get_hyperopt_res(params, train_data, test_data)
