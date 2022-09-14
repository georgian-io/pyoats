"""
Li, Jia and Di, Shimin and Shen, Yanyan and Chen, Lei
"FluxEV: A Fast and Effective Unsupervised Framework for Time-Series Anomaly Detection"
https://doi.org/10.1145/3437963.3441823

Siffer, Alban and Fouque, Pierre-Alain and Termier, Alexandre and Largouet, Christine
"Anomaly Detection in Streams with Extreme Value Theory"
https://doi.org/10.1145/3097983.3098144
"""


import numpy as np
from scipy.stats import genpareto

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
        mu, sigma, gamma = genpareto.fit(peak_set)
        return sigma, gamma

    def __init__(self, **kwargs):
        self._thresholders = None

    def fit(self, data):
        multivar = True if data.ndim > 1 and data.shape[1] > 1 else False
        if multivar:
            self._thresholders = self._pseudo_mv_fit(data)
            return
        return

    def get_threshold(self, data, q=1e-4, contamination=0.95):
        multivar = True if data.ndim > 1 and data.shape[1] > 1 else False
        if multivar:
            if not self._thresholders:
                self._thresholders = self._pseudo_mv_fit(data)
            return self._handle_multivariate(
                data, self._thresholders, q=q, contamination=contamination
            )

        t = self._set_initial_threshold(contamination, data)
        y = self._get_peak_set(t, data)
        n_y = len(y)
        n = len(data)
        sigma, gamma = self._get_gpd_param(y)

        new_threshold = t + sigma / gamma * ((q * n / n_y) ** (-gamma) - 1)
        return np.tile(new_threshold, len(data))
