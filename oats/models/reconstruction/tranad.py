"""
TranAD
-----------------
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from importlib_metadata import version
from oats.models._base import Model
from torch.nn import TransformerDecoder, TransformerEncoder
from torch.utils.data import DataLoader, Dataset, TensorDataset
from oats._utils.dlutils import *
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler


# Proposed Model + Self Conditioning + Adversarial + MAML (TKDE 21)
class _TranAD(nn.Module):
    def __init__(self, feats, window=100):
        super(_TranAD, self).__init__()
        self.name = "TranAD"
        self.lr = 0.005
        self.batch = 128
        self.n_feats = feats
        self.n_window = window
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


class TranADModel(Model):
    """TranAD Model

    Tuli, Shreshth and Casale, Giuliano and Jennings, Nicholas R
    "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data"

    Implementation from reference: https://github.com/imperial-qore/TranAD
    """

    def __init__(
        self,
        window: int = 100,
        use_gpu: bool = False,
        val_split: float = 0.2,
    ):
        """
        Args:
            window (int, optional): rolling window size to feed into the predictor. Defaults to 10.
            use_gpu (bool, optional): whether to use GPU. Defaults to False.
            val_split (float, optional): proportion of data points reserved for validation. Defaults to 0.2.
        """
        # initiate parameters
        self.model_cls = _TranAD
        self.window = window
        self.use_gpu = use_gpu
        self.val_split = val_split

    @property
    def _model_name(self):
        return type(self).__name__

    def _init_model(self, data_dims: int):
        self.model = self.model_cls(data_dims).double()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.model.lr, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)

        return

    def fit(self, train_data, epochs=5):
        self.epochs = epochs
        train_data = train_data if train_data.ndim > 1 else train_data[:, np.newaxis]
        self._init_model(train_data.shape[-1])

        train_data = torch.tensor(train_data)
        trainD = self._convert_to_windows(train_data)

        for e in range(epochs):
            self._backprop(trainD)  # returns loss, lr

        return

    def get_scores(self, test_data):
        test_data = test_data if test_data.ndim > 1 else test_data[:, np.newaxis]
        test_data = torch.tensor(test_data)
        feats = test_data.shape[-1]
        data_len = len(test_data)
        test_data = self._convert_to_windows(test_data)
        dataloader = self._get_dataloader(test_data, data_len)

        l = nn.MSELoss(reduction="none")

        for d, _ in dataloader:
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, data_len, feats)
            z = self.model(window, elem)
            if isinstance(z, tuple):
                z = z[1]
            loss = l(z, elem)[0]

        scores = loss.detach().numpy()

        if scores.shape[1] == 1:
            scores = scores.flatten()
        return scores

    def _convert_to_windows(self, data):
        windows = []
        for i, g in enumerate(data):
            if i >= self.window:
                w = data[i - self.window : i]
            else:
                w = torch.cat([data[0].repeat(self.window - i, 1), data[0:i]])
            windows.append(w)
        windows = torch.stack(windows)
        return windows

    def _get_dataloader(self, data, bs):
        data_x = torch.DoubleTensor(data)
        dataset = TensorDataset(data_x, data_x)

        return DataLoader(dataset, batch_size=bs)

    def _backprop(self, data):
        dataloader = self._get_dataloader(data, self.model.batch)

        feats = data.shape[-1]
        l = nn.MSELoss(reduction="none")
        n = self.epochs + 1

        l1s, l2s = [], []

        for d, _ in dataloader:
            local_bs = d.shape[0]
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, local_bs, feats)
            z = self.model(window, elem)
            l1 = (
                l(z, elem)
                if not isinstance(z, tuple)
                else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
            )
            if isinstance(z, tuple):
                z = z[1]

            l1s.append(torch.mean(l1).item())
            loss = torch.mean(l1)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

        self.scheduler.step()

        return np.mean(l1s), self.optimizer.param_groups[0]["lr"]

    def _scale_series(self, series: npt.NDArray[Any]):
        series = TimeSeries.from_values(series)

        if self.transformer is None:
            self.transformer = Scaler()
            self.transformer.fit(series)

        series = self.transformer.transform(series)

        return series.pd_dataframe().to_numpy().astype(np.float32)
