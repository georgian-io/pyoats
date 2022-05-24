from darts import models

from one.models.predictive.darts_model import DartsModel


class TransformerModel(DartsModel):
    def __init__(self,
                 window: int,
                 n_steps: int,
                 use_gpu: bool,
                 model_type: str = "LSTM",
                 val_split: float = 0.05):

        trainer_kwargs = self._get_trainer_kwargs(use_gpu)
        model = models.TransformerModel(window,
                                        model_type,
                                        pl_trainer_kwargs=trainer_kwargs)

        super().__init__(model, window, n_steps, val_split)
