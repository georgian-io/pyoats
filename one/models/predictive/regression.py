from darts import models

from one.models.darts_simple import SimpleDartsModel


class RegressionModel(SimpleDartsModel):
    def __init__(
        self,
        window: int = 10,
        n_steps: int = 1,
        lags: int = 1,
        val_split: float = 0.2,
        **kwargs
    ):

        model_cls = models.RegressionModel

        super().__init__(model_cls, window, n_steps, lags, val_split)

    def hyperopt_model(self, *args):
        # overriding parent method as there's nothing to tune
        return
