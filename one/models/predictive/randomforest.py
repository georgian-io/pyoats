from darts import models

from one.models.predictive.darts_simple import SimpleDartsModel


class RandomForestModel(SimpleDartsModel):
    def __init__(self,
                 window: int = 10,
                 n_steps: int = 1,
                 lags: int = 1):

        model = models.RandomForest

        super().__init__(model, window, n_steps, lags)

