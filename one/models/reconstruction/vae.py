from pyod.models.vae import VAE
from one.models.pyod_model import PyODModel


class VAEModel(PyODModel):
    def __init__(self, window=10, **kwargs):
        model_cls = VAE
        super().__init__(model_cls, window, **kwargs)