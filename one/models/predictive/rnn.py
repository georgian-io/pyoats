from typing import Literal
from functools import partialmethod

from darts import models

from one.models.predictive.darts_model import DartsModel


class RNNModel(DartsModel):
    def __init__(self,
                 window: int = 10,
                 n_steps: int = 1,
                 use_gpu: bool = False,
                 val_split: float = 0.05,
                 rnn_model: str = "RNN"):

        model_cls = models.RNNModel

        super().__init__(model_cls, window, n_steps, use_gpu, val_split, rnn_model = rnn_model)
