from tods.sk_interface.detection_algorithm.IsolationForest_skinterface import (
    IsolationForestSKI,
)

from one.models.tods_model import TODSModel


class IsolationForestModel(TODSModel):
    def __init__(self, window: int = 10, n_steps: int = 1):

        model_cls = IsolationForestSKI

        super().__init__(model_cls, window, n_steps)

    def hyperopt_model(self, *args):
        # overriding parent method as there's nothing to tune
        return
