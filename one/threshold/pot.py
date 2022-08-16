import numpy as np

from one.threshold.base import Threshold


class POTThreshold(Threshold):
    @classmethod
    def _set_initial_threshold(cls, contamination: float, scores) -> float:
        return np.quantile(scores, contamination)

    @classmethod
    def _get_peak_set(cls, t: float, scores):
        x = scores.copy()
        x = x[x >= t]

        if len(x) == 0:
            t *= 0.95
            t = 0 if t < 1e-3 else t
            return cls._get_peak_set(t, scores)

        return x - t

    @classmethod
    def _get_gpd_param(cls, peak_set):
        y = peak_set.copy()
        mu = y.mean()
        var_y = y.var(ddof=1)

        if var_y == 0: return 0, 1

        sigma = mu/2 * (1 + mu ** 2 / var_y)
        gamma = 1/2 * (1 - mu ** 2 / var_y)

        return sigma, gamma

    def __init__(self, **kwargs):
        self._thresholders = None


    def fit(self, data):
        multivar = True if data.ndim > 1 and data.shape[1] > 1 else False
        if multivar:
            self._thresholders = self._pseudo_mv_fit(data)
            return
        return

    def get_threshold(self, data, q=1e-4, contamination = 0.95):
        multivar = True if data.ndim > 1 and data.shape[1] > 1 else False
        if multivar:
            if not self._thresholders: self._thresholders = self._pseudo_mv_fit(data)
            return self._handle_multivariate(data, self._thresholders, q=q, contamination=contamination)
 
        t = self._set_initial_threshold(contamination, data)
        y = self._get_peak_set(t, data)
        n_y = len(y)
        n = len(data)
        sigma, gamma = self._get_gpd_param(y)

        new_threshold = t + sigma/gamma * ((q*n/n_y)**(-gamma) - 1)
        return np.tile(new_threshold, len(data))