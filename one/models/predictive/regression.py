from darts import models

from one.models.predictive.darts_simple import SimpleDartsModel


class RegressionModel(SimpleDartsModel):
    def __init__(self, lags: int = -1):

        model = models.RegressionModel(lags)

        #TODO: OVERRIDE MOETHODS!!

