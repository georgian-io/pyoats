from typing import Any, Tuple

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler

from one.models.base import Model



class SimpleDartsModel(Model):
    pass
