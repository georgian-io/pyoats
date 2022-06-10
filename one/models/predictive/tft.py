from typing import Any
from functools import partial

from darts import models
import numpy as np
import numpy.typing as npt
import optuna

from one.models.predictive.darts_model import DartsModel


class TFTModel(DartsModel):
    def __init__(
        self,
        window: int = 10,
        n_steps: int = 1,
        use_gpu: bool = 1,
        val_split: float = 0.2,
    ):

        model = models.TFTModel

        super().__init__(model, window, n_steps, use_gpu, val_split)

    def _model_objective(self, trial, train_data: npt.NDArray[Any]):
        params = {
            "add_relative_index": trial.suggest_categorical(
                "add_relative_idex", [True]
            ),
            "hidden_size": trial.suggest_int("hidden_size", 8, 128),
            "lstm_layers": trial.suggest_int("lstm_layers", 1, 32),
            "num_attention_heads": trial.suggest_int("num_attention_heads", 2, 8),
            "hidden_continuous_size": trial.suggest_int(
                "hidden_continuous_size", 4, 32
            ),
            "full_attention": trial.suggest_categorical(
                "full_attention", [True, False]
            ),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
       }

        return self._get_hyperopt_res(params, train_data)
