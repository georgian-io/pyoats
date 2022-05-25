from darts import models

from one.models.predictive.darts_model import DartsModel


class NBEATSModel(DartsModel):
    def __init__(
        self,
        window: int = 10,
        n_steps: int = 1,
        use_gpu: bool = False,
        val_split: float = 0.05,
    ):

        model = models.NBEATSModel

        super().__init__(model, window, n_steps, val_split)
