from darts import models

from one.models.predictive.darts_simple import SimpleDartsModel


class LightGBMModel(SimpleDartsModel):
    def __init__(self, window: int = 10, n_steps: int = 1, lags: int = 1):

        model_cls = models.LightGBMModel

        super().__init__(model_cls, window, n_steps, lags)
