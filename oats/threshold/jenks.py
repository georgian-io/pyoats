"""
Jenks Natural Breaks
-----------------
"""

import numpy as np
import jenkspy

from oats.threshold._base import Threshold


class JenksThreshold(Threshold):
    """Getting threshold using 1-D clustering (Jenks Natural Breaks).
    No need to call `fit()`.
    """

    def fit(self, *args, **kwargs):
        return

    def get_threshold(self, data, n_partitions=20):
        """
        Args:
            data (np.ndarray): numpy array of scores
            n_partitions (int, optional): number of partitions to fit. Defaults to 20.

        Returns:
            np.ndarray: thresholds
        """
        multivar = True if data.ndim > 1 and data.shape[1] > 1 else False
        if multivar:
            return self._handle_multivariate(
                data, [self] * data.shape[1], n_partitions=n_partitions
            )
        thres = jenkspy.jenks_breaks(data.flatten(), nb_class=n_partitions)[-2]
        return np.tile(thres, len(data))
