from typing import Any
from functools import partial

from darts import models
import numpy as np
import numpy.typing as npt
import optuna

from one.models.predictive.darts_model import DartsModel


class NHiTSModel(DartsModel):
    def __init__(
        self,
        window: int = 10,
        n_steps: int = 1,
        use_gpu: bool = False,
        val_split: float = 0.05,
    ):

        model_cls = models.NHiTS

        super().__init__(model_cls, window, n_steps, use_gpu, val_split)

    def _model_objective(
        self, trial, train_data: npt.NDArray[Any], test_data: npt.NDArray[Any]
    ):
        params = {
            "num_stacks": trial.suggest_int("num_stacks", 1, 16),
            "num_blocks": trial.suggest_int("num_blocks", 1, 16),
            "num_layers": trial.suggest_int("num_layers", 1, 32),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "optimizer_kwargs": {"lr": trial.suggest_uniform("lr", 1e-5, 1e-1)},
            "batch_size": trial.suggest_int(
                "min_child_samples", 1, (len(train_data) - self.window) // self.n_steps
            ),
        }

        self.model = self._init_model(**params)
        self.fit(train_data)
        _, res, _ = self.get_scores(test_data)

        return np.sum(res**2)
