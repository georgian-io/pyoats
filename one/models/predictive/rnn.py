from typing import Any, Literal
from functools import partial

from darts import models
import numpy as np
import numpy.typing as npt
import optuna

from one.models.predictive.darts_model import DartsModel


class RNNModel(DartsModel):
    def __init__(
        self,
        window: int = 10,
        n_steps: int = 1,
        use_gpu: bool = False,
        val_split: float = 0.05,
        rnn_model: str = "RNN",
    ):

        model_cls = models.RNNModel

        super().__init__(
            model_cls, window, n_steps, use_gpu, val_split, rnn_model=rnn_model
        )

    def hyperopt_model(
        self,
        train_data: npt.NDArray[Any],
        test_data: npt.NDArray[Any],
        n_trials: int = 30,
    ):
        # TODO: we can probably merge this with the hyperparam tuning method in the parent...

        # obj
        obj = partial(
            self._model_objective,
            train_data=train_data,
            test_data=test_data,
        )

        study = optuna.create_study()
        study.optimize(obj, n_trials=n_trials)

        self.params = study.best_params
        self.model = self._init_model(**self.params)

    def _model_objective(
        self, trial, train_data: npt.NDArray[Any], test_data: npt.NDArray[Any]
    ):
        params = {
            "hidden_dim": trial.suggest_int("hidden_dim", 10, 256),
            "n_rnn_layers": trial.suggest_int("n_rnn_layers", 1, 64),
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
