from darts import models

from one.models.predictive.darts_model import DartsModel


class NHiTSModel(DartsModel):
    def __init__(self,
                 window: int = 10,
                 n_steps: int = 1,
                 use_gpu: bool = False,
                 val_split: float = 0.05):

        model_cls = models.NHiTS

        super().__init__(model_cls, window, n_steps, use_gpu, val_split)
