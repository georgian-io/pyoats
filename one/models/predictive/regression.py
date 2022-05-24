from darts import models

from one.models.predictive.darts_simple import SimpleDartsModel


class RegressionModel(SimpleDartsModel):
    def __init__(self,
                 window: int,
                 n_steps: int,
                 lags: int = 1):

        model = models.RegressionModel(lags)

        super().__init__(model, window, n_steps)


