"""
Transformer
-----------------
"""
from typing import Any
from functools import partial

from darts import models
import numpy as np
import numpy.typing as npt
import optuna


from oats.models._darts_model import DartsModel


class TransformerModel(DartsModel):
    """Transformer Model

    Using Transformer as a predictor. Anomalies scores are deviations from predictions.

    Reference: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.transformer_model.html
    """

    def __init__(
        self,
        window: int = 10,
        n_steps: int = 1,
        use_gpu: bool = False,
        val_split: float = 0.2,
    ):
        """
        initialization also accepts any parameters used by: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.transformer_model.html

        Args:
            window (int, optional): rolling window size to feed into the predictor. Defaults to 10.
            n_steps (int, optional): number of steps to predict forward. Defaults to 1.
            use_gpu (bool, optional): whether to use GPU. Defaults to False.
            val_split (float, optional): proportion of data points reserved for validation. Defaults to 0.2.
        """

        model = models.TransformerModel
        super().__init__(model, window, n_steps, use_gpu, val_split)

    def _model_objective(self, trial, train_data: npt.NDArray[Any]):
        """
        params = {
            # "nhead": trial.suggest_int("nhead", 2, 8, 2),
        }

        dep_params = {
            "dim_feedforward": trial.suggest_int("dim_feedforward", 256, 1024),
            "num_encoder_layers": trial.suggest_int("num_encoder_layers", 2, 10),
            "num_decoder_layers": trial.suggest_int("num_decoder_layers", 2, 10),
            "d_model": trial.suggest_int("d_model", 32, 256),
        }

        params.update(dep_params)
        """

        # TODO: figure out why this isn't working
        return 0
