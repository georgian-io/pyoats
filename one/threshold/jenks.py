import numpy as np
import jenkspy

from one.threshold.base import Threshold


class JenksThreshold(Threshold):
    def fit(self, *args, **kwargs):
        return

    def get_threshold(self, data, n_partitions=20):
        multivar = True if data.ndim > 1 and data.shape[1] > 1 else False
        if multivar:
            return self._handle_multivariate(
                data, [self] * data.shape[1], n_partitions=n_partitions
            )
        thres = jenkspy.jenks_breaks(data, nb_class=n_partitions)[-2]
        return np.tile(thres, len(data))
