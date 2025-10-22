"""
TSMixer
-----------------
"""

from typing import Any

import numpy.typing as npt
from darts import models

from oats.models._darts_model import DartsModel


class TSMixerModel(DartsModel):
    """TSMixer Model

    An MLP based model that combines temporal, static and cross-sectional feature
    information using stacked mixing layers.

    Using TSMixer as a predictor. Anomaly scores are deviations from predictions.

    Reference: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tsmixer_model.html
    Paper: https://arxiv.org/abs/2303.06053
    """

    def __init__(
        self,
        window: int = 10,
        n_steps: int = 1,
        use_gpu: bool = False,
        val_split: float = 0.2,
        **kwargs,
    ):
        """
        Initialization also accepts any parameters used by:
        https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tsmixer_model.html

        Args:
            window (int, optional): rolling window size to feed into the predictor. Defaults to 10.
            n_steps (int, optional): number of steps to predict forward. Defaults to 1.
            use_gpu (bool, optional): whether to use GPU. Defaults to False.
            val_split (float, optional): proportion of data points reserved for validation. Defaults to 0.2.
            **kwargs: additional parameters for TSMixerModel (hidden_size, ff_size, num_blocks,
                      activation, dropout, norm_type, etc.)
        """
        model_cls = models.TSMixerModel

        super().__init__(model_cls, window, n_steps, use_gpu, val_split, **kwargs)

    def _model_objective(self, trial, train_data: npt.NDArray[Any]):
        """
        Hyperparameter optimization objective function.

        Example hyperparameters to optimize:
        params = {
            "hidden_size": trial.suggest_int("hidden_size", 32, 128),
            "ff_size": trial.suggest_int("ff_size", 32, 128),
            "num_blocks": trial.suggest_int("num_blocks", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "batch_size": trial.suggest_int(
                "batch_size", 1, (len(train_data) - self.window) // self.n_steps // 4
            ),
        }
        """

        # return self._get_hyperopt_res(params, train_data)
        return 0
