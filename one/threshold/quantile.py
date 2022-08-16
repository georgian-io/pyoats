import numpy as np

from one.threshold.base import Threshold

class QuantileThreshold(Threshold):
    def fit(self, *args, **kwargs):
        return 

    def get_threshold(self, data, percentile=0.95):
        multivar = True if data.ndim > 1 and data.shape[1] > 1 else False
        if multivar:
            tile = (len(data), 1)
        else: tile = len(data)
            
        return np.tile(np.quantile(data, percentile, axis=0), tile)