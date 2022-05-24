from darts import models

from one.models.predictive.darts_simple import SimpleDartsModel


class RandomForestModel(SimpleDartsModel):
    def __init__(self,
                 window: int,
                 n_steps: int,
                 lags: int = 1):

        model = models.RandomForest(lags)

        super().__init__(model, window, n_steps)

