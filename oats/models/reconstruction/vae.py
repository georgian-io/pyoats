"""
Variational Auto-Encoder (VAE)
-----------------
"""

from pyod.models.vae import VAE

from oats.models._pyod_model import PyODModel


class VAEModel(PyODModel):
    """VAE Model (Variational Auto-encover)

    Using reconstruction error of the trained encoder-decoder network as anomaly scores.

    Reference: https://pyod.readthedocs.io/en/latest/_modules/pyod/models/vae.html
    """

    def __init__(self, window=10, **kwargs):
        """
        initialization also accepts any parameters used by: https://pyod.readthedocs.io/en/latest/_modules/pyod/models/vae.html

        Args:
            window (int, optional): _description_. Defaults to 10.
        """
        model_cls = VAE
        self.window = window

        super().__init__(model_cls, window, **kwargs)

    def fit(self, train_data, **kwargs):
        n_feat = (
            train_data.shape[1]
            if train_data.ndim > 1 and train_data.shape[1] > 1
            else 1
        )

        if not self.params.get("encoder_neuron_list"):
            self.params["encoder_neuron_list"] = [
                n_feat * self.window,
                n_feat * self.window // 2,
                n_feat * self.window // 4,
            ]
        if not self.params.get("decoder_neuron_list"):
            self.params["decoder_neuron_list"] = [
                n_feat * self.window // 4,
                n_feat * self.window // 2,
                n_feat * self.window,
            ]
        self.model = VAE(**self.params)

        super().fit(train_data, **kwargs)
