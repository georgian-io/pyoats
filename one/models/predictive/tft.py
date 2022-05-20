from darts import models

from one.models.predictive.darts_model import DartsModel


class TCNModel(DartsModel):
    def __init__(self,
                 window: int,
                 n_steps: int,
                 use_gpu: bool,
                 val_split: float = 0.05):

        trainer_kwargs = self._get_trainer_kwargs(use_gpu)
        model = models.TFTModel(window,
                            n_steps,
                            pl_trainer_kwargs=trainer_kwargs
                            )

        super().__init__(model, window, n_steps, val_split)
