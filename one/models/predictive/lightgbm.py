from darts import models

from one.models.predictive.darts_simple import SimpleDartsModel


class LightGBMModel(SimpleDartsModel):
    def __init__(self, lags: int = -1):

        model = models.LightGBMModel(lags)

        #TODO: OVERRIDE MOETHODS!!

