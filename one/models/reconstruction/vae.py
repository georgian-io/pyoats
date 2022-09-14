from pyod.models.vae import VAE
from one.models.pyod_model import PyODModel


class VAEModel(PyODModel):
    def __init__(self, window=10, **kwargs):
        self.IS_DL = False

        model_cls = VAE
        self.window = window

        super().__init__(model_cls, window, **kwargs)

    def fit(self, train_data, **kwargs):
        n_feat = (
            train_data.shape[1]
            if train_data.ndim > 1 and train_data.shape[1] > 1
            else 1
        )

        if not self.params.get("encoder_neurons"):
            self.params["encoder_neurons"] = [
                n_feat * self.window,
                n_feat * self.window // 2,
                n_feat * self.window // 4,
            ]
        if not self.params.get("decoder_neurons"):
            self.params["decoder_neurons"] = [
                n_feat * self.window // 4,
                n_feat * self.window // 2,
                n_feat * self.window,
            ]
        self.model = VAE(**self.params)

        super().fit(train_data, **kwargs)
