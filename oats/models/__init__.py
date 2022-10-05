"""
Models
-----------------
"""

import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Predictive Models
from oats.models.predictive.lightgbm import LightGBMModel
from oats.models.predictive.nbeats import NBEATSModel
from oats.models.predictive.nhits import NHiTSModel
from oats.models.predictive.randomforest import RandomForestModel
from oats.models.predictive.rnn import RNNModel
from oats.models.predictive.tcn import TCNModel
from oats.models.predictive.tft import TFTModel
from oats.models.predictive.transformer import TransformerModel
from oats.models.predictive.ma import MovingAverageModel
from oats.models.predictive.arima import ARIMAModel
from oats.models.predictive.regression import RegressionModel
from oats.models.predictive.fluxev import FluxEVModel

# Reconstruction
from oats.models.reconstruction.tranad import TranADModel
from oats.models.reconstruction.vae import VAEModel

# Distance-Based
from oats.models.distance.iforest import IsolationForestModel
from oats.models.distance.matrixprofile import MatrixProfileModel

# Rule-Based
from oats.models.rule_based.quantile import QuantileModel
