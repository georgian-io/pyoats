"""
Recurrent Neural Networks (RNN)
-----------------
"""
from typing import Any, Literal
from functools import partial

from darts import models
import numpy as np
import numpy.typing as npt
import optuna

from oats.models._darts_model import DartsModel


class RNNModel(DartsModel):
    """Recurrent Neural Network Model

    Using RNN as a predictor. Anomalies scores are deviations from predictions.

    Reference: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.rnn_model.html
    """

    def __init__(
        self,
        window: int = 10,
        n_steps: int = 1,
        use_gpu: bool = False,
        val_split: float = 0.2,
        rnn_model: str = "RNN",
        **kwargs
    ):
        """
        initialization also accepts any parameters used by: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.rnn_model.html

        Args:
            window (int, optional): rolling window size to feed into the predictor. Defaults to 10.
            n_steps (int, optional): number of steps to predict forward. Defaults to 1.
            use_gpu (bool, optional): whether to use GPU. Defaults to False.
            val_split (float, optional): proportion of data points reserved for validation. Defaults to 0.2.
            rnn_model (str, optional): `RNN` (vanilla RNN), `LSTM`, or `GRU`. Defaults to `RNN`.
        """

        model_cls = models.RNNModel

        super().__init__(
            model_cls,
            window,
            n_steps,
            use_gpu,
            val_split,
            rnn_model=rnn_model,
            **kwargs
        )

    def _model_objective(self, trial, train_data: npt.NDArray[Any]):
        params = {
            "hidden_dim": trial.suggest_int("hidden_dim", 10, 256),
            "n_rnn_layers": trial.suggest_int("n_rnn_layers", 1, 64),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "batch_size": trial.suggest_int(
                "batch_size", 1, (len(train_data) - self.window) // self.n_steps // 4
            ),
        }

        return self._get_hyperopt_res(params, train_data)
