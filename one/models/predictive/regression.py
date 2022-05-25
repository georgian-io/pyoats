from darts import models

from one.models.predictive.darts_simple import SimpleDartsModel


class RegressionModel(SimpleDartsModel):
    def __init__(self, window: int = 10, n_steps: int = 1, lags: int = 1):

        model_cls = models.RegressionModel

        super().__init__(model_cls, window, n_steps, lags)

    def hyperopt_model(self):
        # overriding parent method as there's nothing to tune
        return
