"""
Peaks-Over-Threshold (POT)
-----------------
"""

import numpy as np
from scipy.stats import genpareto

from oats.threshold._base import Threshold


class POTThreshold(Threshold):
    """Fit the tails of the data with Generalized Pareto Distribution (GPD).
    Find the threshold where `P(thres) < q`.
    Usual values for q is 1e-3 to 1e-6.

    Siffer, Alban and Fouque, Pierre-Alain and Termier, Alexandre and Largouet, Christine
    "Anomaly Detection in Streams with Extreme Value Theory"
    https://doi.org/10.1145/3097983.3098144
    """

    @classmethod
    def _set_initial_threshold(cls, tail_level: float, scores) -> float:
        return np.quantile(scores, tail_level)

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

    def get_threshold(self, data, q: float = 1e-4, tail_level: float = 0.95):
        """
        Args:
            data (np.ndarray): array of data/anomaly scores
            q (float, optional): q level such that `P(threshold) < q`. Defaults to 1e-4.
            tail_level (float, optional): threshold to fit tail distribution. Defaults to 0.95.

        Returns:
            np.ndarray: array of thresholds
        """
        multivar = True if data.ndim > 1 and data.shape[1] > 1 else False
        if multivar:
            if not self._thresholders:
                self._thresholders = self._pseudo_mv_fit(data)
            return self._handle_multivariate(
                data, self._thresholders, q=q, tail_level=tail_level
            )

        t = self._set_initial_threshold(tail_level, data)
        y = self._get_peak_set(t, data)
        n_y = len(y)
        n = len(data)
        sigma, gamma = self._get_gpd_param(y)

        new_threshold = t + sigma / gamma * ((q * n / n_y) ** (-gamma) - 1)
        return np.tile(new_threshold, len(data))
