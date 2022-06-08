from typing import Any
from functools import partial

from darts import models
import numpy as np
import numpy.typing as npt
import optuna


from one.models.predictive.darts_model import DartsModel


class TransformerModel(DartsModel):
    def __init__(
        self,
        window: int = 10,
        n_steps: int = 1,
        use_gpu: bool = False,
        val_split: float = 0.2,
    ):

        model = models.TransformerModel
        super().__init__(model, window, n_steps, use_gpu, val_split)

    def _model_objective(self, trial, train_data: npt.NDArray[Any]):
        params = {
            # "nhead": trial.suggest_int("nhead", 2, 8, 2),
        }

        dep_params = {
            "dim_feedforward": trial.suggest_int("dim_feedforward", 256, 1024),
            "num_encoder_layers": trial.suggest_int("num_encoder_layers", 2),
            "num_decoder_layers": trial.suggest_int("num_decoder_layers", 2),
            "d_model": trial.suggest_int("d_model", 32, 256, params["nhead"]),
        }

        params.update(dep_params)

        # TODO: figure out why this isn't working
        return
