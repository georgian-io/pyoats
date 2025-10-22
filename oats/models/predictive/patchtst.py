"""
PatchTST
-----------------
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from transformers import PatchTSTConfig, PatchTSTForPrediction

from oats.models._base import Model


class PatchTSTModel(Model):
    """PatchTST Model - Patch-based Transformer for Time Series

    A time series is worth 64 words: Long-term forecasting with transformers.
    PatchTST segments time series into subseries-level patches which serve as input
    tokens to a Transformer, achieving state-of-the-art performance with reduced
    computational complexity.

    Implementation using Hugging Face Transformers library.

    Reference: https://huggingface.co/docs/transformers/model_doc/patchtst
    Paper: https://arxiv.org/abs/2211.14730
    """

    def __init__(
        self,
        window: int = 512,
        n_steps: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        use_gpu: bool = False,
        d_model: int = 128,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 3,
        ffn_dim: int = 512,
        dropout: float = 0.2,
        **kwargs,
    ):
        """
        Args:
            window (int, optional): Context length (look-back window). Defaults to 512.
            n_steps (int, optional): Prediction horizon (forecast length). Defaults to 96.
            patch_len (int, optional): Length of each patch. Defaults to 16.
            stride (int, optional): Stride between patches. Defaults to 8.
            use_gpu (bool, optional): Whether to use GPU. Defaults to False.
            d_model (int, optional): Dimensionality of the model. Defaults to 128.
            num_attention_heads (int, optional): Number of attention heads. Defaults to 4.
            num_hidden_layers (int, optional): Number of encoder layers. Defaults to 3.
            ffn_dim (int, optional): Dimension of feedforward network. Defaults to 512.
            dropout (float, optional): Dropout probability. Defaults to 0.2.
            **kwargs: Additional parameters passed to PatchTSTConfig.
        """
        self.window = window
        self.n_steps = n_steps
        self.patch_len = patch_len
        self.stride = stride
        self.use_gpu = use_gpu
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        # Determine device
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        # Initialize model as None (will be created in fit based on input channels)
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None

    @property
    def _model_name(self):
        return type(self).__name__

    def __repr__(self):
        r = {}
        r.update({"model_name": self._model_name})
        r.update({"window": self.window})
        r.update({"n_steps": self.n_steps})
        r.update({"patch_len": self.patch_len})
        r.update({"stride": self.stride})
        r.update({"d_model": self.d_model})
        return str(r)

    def _init_model(self, num_channels: int):
        """Initialize the PatchTST model with the given number of channels."""
        config = PatchTSTConfig(
            prediction_length=self.n_steps,
            context_length=self.window,
            patch_length=self.patch_len,
            stride=self.stride,
            num_input_channels=num_channels,
            d_model=self.d_model,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
        )
        self.model = PatchTSTForPrediction(config).to(self.device)

    def _normalize(self, data: npt.NDArray[Any], fit: bool = False):
        """Normalize data using z-score normalization."""
        if fit:
            self.scaler_mean = np.mean(data, axis=0, keepdims=True)
            self.scaler_std = np.std(data, axis=0, keepdims=True) + 1e-8

        return (data - self.scaler_mean) / self.scaler_std

    def _denormalize(self, data: npt.NDArray[Any]):
        """Denormalize data."""
        return data * self.scaler_std + self.scaler_mean

    def fit(self, train_data: npt.NDArray[Any], epochs: int = 10, **kwargs):
        """
        Train the PatchTST model.

        Args:
            train_data: Training time series data
            epochs: Number of training epochs (default: 10)
        """
        # Handle different input shapes
        if train_data.ndim == 1:
            train_data = train_data[:, np.newaxis]  # (t,) -> (t, 1)

        num_channels = train_data.shape[1]

        # Initialize model if not already done
        if self.model is None:
            self._init_model(num_channels)

        # Normalize data
        train_data_norm = self._normalize(train_data, fit=True).astype(np.float32)

        # Prepare training data as tensors
        train_tensor = torch.tensor(train_data_norm, dtype=torch.float32).to(self.device)

        # Create simple training loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()

        for epoch in range(epochs):
            # Create sliding windows for training
            for i in range(0, len(train_tensor) - self.window - self.n_steps, self.n_steps):
                context = train_tensor[i : i + self.window].unsqueeze(0)  # (1, window, channels)
                target = train_tensor[i + self.window : i + self.window + self.n_steps, :]  # (n_steps, channels)

                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(past_values=context)
                predictions = outputs.prediction_outputs[:, :, :]  # (1, n_steps, channels)

                # Compute loss
                loss = torch.nn.functional.mse_loss(predictions.squeeze(0), target)

                # Backward pass
                loss.backward()
                optimizer.step()

    def get_scores(self, test_data: npt.NDArray[Any], **kwargs):
        """
        Generate anomaly scores for test data.

        Args:
            test_data: Test time series data

        Returns:
            Anomaly scores (absolute prediction residuals)
        """
        # Handle different input shapes
        original_shape = test_data.shape
        if test_data.ndim == 1:
            test_data = test_data[:, np.newaxis]

        multivar = test_data.shape[1] > 1

        # Normalize
        test_data_norm = self._normalize(test_data).astype(np.float32)
        test_tensor = torch.tensor(test_data_norm, dtype=torch.float32).to(self.device)

        self.model.eval()
        all_residuals = []

        with torch.no_grad():
            # Generate predictions using sliding windows
            for i in range(0, len(test_tensor) - self.window, self.n_steps):
                context = test_tensor[i : i + self.window].unsqueeze(0)

                # Predict
                outputs = self.model(past_values=context)
                predictions = outputs.prediction_outputs.squeeze(0)  # (n_steps, channels)

                # Get actual values
                end_idx = min(i + self.window + self.n_steps, len(test_tensor))
                actual_len = end_idx - (i + self.window)
                actual = test_tensor[i + self.window : end_idx, :]

                # Compute residuals
                residuals = torch.abs(predictions[:actual_len, :] - actual)
                all_residuals.append(residuals.cpu().numpy())

        # Concatenate all residuals
        if all_residuals:
            scores = np.concatenate(all_residuals, axis=0)

            # Pad the beginning with zeros (for the initial window)
            padding = np.zeros((self.window, test_data.shape[1]), dtype=np.float32)
            scores = np.vstack([padding, scores])

            # Trim or pad to match input length
            if len(scores) > len(test_data):
                scores = scores[:len(test_data), :]
            elif len(scores) < len(test_data):
                # Pad end with mean score
                pad_len = len(test_data) - len(scores)
                pad_val = np.mean(scores, axis=0, keepdims=True)
                padding_end = np.repeat(pad_val, pad_len, axis=0)
                scores = np.vstack([scores, padding_end])
        else:
            # Fallback if no predictions were made
            scores = np.zeros_like(test_data, dtype=np.float32)

        # Return shape consistent with OATS conventions
        # Univariate: always return (t,) regardless of input shape (t,) or (t,1)
        # Multivariate: return (t, n)
        if len(original_shape) == 1 or (len(original_shape) == 2 and original_shape[1] == 1):
            return scores.flatten()  # (t,) or (t, 1) -> (t,)
        else:
            return scores  # (t, n)
