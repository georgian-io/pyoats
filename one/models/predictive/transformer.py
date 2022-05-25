from darts import models

from one.models.predictive.darts_model import DartsModel


class TransformerModel(DartsModel):
    def __init__(
        self, window: int, n_steps: int, use_gpu: bool, val_split: float = 0.05
    ):

        model = models.TransformerModel
        super().__init__(model, window, n_steps, use_gpu, val_split)
