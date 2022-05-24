from darts import models

from one.models.predictive.darts_model import DartsModel


class TCNModel(DartsModel):
    def __init__(self,
                 window: int = 10,
                 n_steps: int = 1,
                 use_gpu: bool = 1,
                 val_split: float = 0.05):

        model = models.TFTModel

        super().__init__(model, window, n_steps, use_gpu, val_split)
