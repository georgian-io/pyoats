"""
Quantile
-----------------
"""
import numpy as np

from oats.threshold._base import Threshold


class QuantileThreshold(Threshold):
    """Simple threshold method where top `q` deciles of anomaly scores are labeled as anomalies.
    Beware that user is guaranteed to have outlier predictions with this method!

    `fit()` not necessary.
    """

    def fit(self, *args, **kwargs):
        return

    def get_threshold(self, data, percentile: float = 0.95):
        """
        Args:
            data (np.ndarray): array of anomaly scores
            percentile (float, optional): decile level used for threshold. Defaults to 0.95.

        Returns:
            np.ndarray: array of thresholds
        """
        multivar = True if data.ndim > 1 and data.shape[1] > 1 else False
        if multivar:
            tile = (len(data), 1)
        else:
            tile = len(data)

        return np.tile(np.quantile(data, percentile, axis=0), tile)
