import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from importlib_metadata import version
from one.models.base import Model
from torch.nn import TransformerDecoder, TransformerEncoder
from torch.utils.data import DataLoader, Dataset, TensorDataset
from utils.dlutils import *


# Proposed Model + Self Conditioning + Adversarial + MAML (TKDE 21)
class TranAD(nn.Module):
    def __init__(self, feats, window=100):
        super(TranAD, self).__init__()
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
    def __init__(
        self,
        model_cls=TranAD,
        window: int = 100,
        n_steps: int = None,  # Not necessary
        use_gpu: bool = False,
        val_split: float = 0.2,
        epochs: int = 5,
    ):

        # initiate parameters
        self.model_cls = model_cls
        self.window = model_cls
        self.n_steps = n_steps
        self.use_gpu = use_gpu
        self.val_split = val_split
        self.epochs = epochs

    @property
    def model_name(self):
        return type(self).__name__

    def _init_model(self, data_dims: int):
        self.model = self.model_cls(
            data_dims,
        ).double()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.model.lr, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)

        return

    def fit(self, train_data):
        self._init_model(train_data.shape[1])
        trainD = self._convert_to_windows(train_data)

        for e in range(self.epochs):
            self._backprop(trainD)  # returns loss, lr

        return

    def get_scores(self, test_data):
        feats = test_data.shape[0]
        data_len = len(test_data)
        test_data = self._convert_to_windows(test_data)
        dataloader = self._get_dataloader(test_data, data_len)

        l = nn.MSELoss(reduction="none")

        for d, _ in dataloader:
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, l, feats)
            z = self.model(window, elem)
            if isinstance(z, tuple):
                z = z[1]
            loss = l(z, elem)[0]

        return loss.detach().numpy(), z.detach().numpy()[0]

    def _convert_to_windows(self, data):
        windows = []
        for i, g in enumerate(data):
            if i >= self.window:
                w = data[i - self.window : i]
            else:
                w = torch.cat([data[0].repeat(self.window - i, 1), data[0:i]])
            windows.append(w)
        return torch.stack(windows)

    def _get_dataloader(self, data, bs):
        data_x = torch.DoubleTensor(data)
        dataset = TensorDataset(data_x, data_x)

        bs = self.model.batch
        return DataLoader(dataset, batch_size=bs)

    def _backprop(self, data):
        dataloader = self._get_dataloader(self, data, self.model.bs)

        feats = data.shape[1]
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
