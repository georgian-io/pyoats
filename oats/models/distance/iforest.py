"""
Isolation Forest
-----------------
"""

from pyod.models.iforest import IForest
from oats.models._pyod_model import PyODModel


class IsolationForestModel(PyODModel):
    def __init__(self, window=10, **kwargs):

        model_cls = IForest

        super().__init__(model_cls, window, **kwargs)

    def hyperopt_model(self, *args):
        # overriding parent method as there's nothing to tune
        return
