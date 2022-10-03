"""
Qualitative Metrics
-----------------
"""

import numpy as np
import scipy.signal as signal

from oats.scorer._base import Scorer


class QualitativeMetrics(Scorer):
    """Unsupervised qualitative metrics used to access the quality of the anomaly detection algorithm"""

    def __init__(self, window=10):
        """
        Args:
            window (int, optional): Window sized used to compute `diff_mean_trend` and `diff_mid_avg`. Defaults to 10.
        """
        self.data = np.array([])
        self.preds = np.array([])

        self.window = window

    def process(self, data, preds):
        if data.ndim > 1 and data.shape[1] == 1:
            data = data.flatten()

        if len(data) != len(preds):
            raise ValueError(
                f"Data of length {len(data)} does not match preds of length {len(preds)}"
            )

        existing_n_feat = self.data.shape[1] if self.data.ndim > 1 else 1
        incoming_n_feat = data.shape[1] if data.ndim > 1 else 1
        if len(self.data) > 0 and existing_n_feat != incoming_n_feat:
            raise ValueError(
                f"Unable to process incoming data of shape {data.shape} with existing data of shape {self.data.shape}"
            )

        if data.ndim > 1 and data.shape[1] > 1:
            self.data = np.vstack([self.data] * data.shape[1]).T

        self.data = np.append(self.data, data, axis=0)
        self.preds = np.append(self.preds, preds)

    @property
    def num_anom(self):
        """Number of predicted anomalies, should be low (as anomalies are rare)"""
        return self.preds.sum()

    @property
    def pct_anom(self):
        """Percentage of predicted anomalies, should be low (as anomalies are rare)"""
        return self.num_anom / len(self.preds)

    @property
    def _pred_anomalies(self):
        return self.data[self.preds == 1]

    @property
    def _pred_non_anomalies(self):
        return self.data[self.preds == 0]

    @property
    def avg_anom_dist_from_mean(self):
        """Distance of predicted anomalies to the mean of original data, should be high; useful for series with a lot of global point anomalies"""
        return np.abs(self._pred_anomalies - self.data.mean(axis=0)).mean()

    @property
    def avg_cycles_delta_between_anom(self):
        """Average time between anomalies, should be high, as anomalies should be occuring far apart (for point anomalies)"""
        if self.num_anom in (0, 1, len(self.data)):
            return 0
        return np.diff(np.where(self.preds == 1)[0]).mean()

    @property
    def max_range_non_anom(self):
        """The tightness of data from predicted non-anomalies, similar to the idea of `avg_anom_dist_from_mean`; should be low"""
        if self.num_anom in (0, len(self.data)):
            return 1e5
        return (
            np.abs(self._pred_non_anomalies.max() - self._pred_non_anomalies.min())
        ).mean()

    def _get_mid_avg_filter(self):
        # make sure window is odd
        if self.window % 2 == 0:
            self.window += 1
        padding = self.window // 2

        # local difference filter
        fil = self.window
        fil = np.full((self.window), -1 / (self.window - 1))
        fil[padding] = 1
        # if self.data.ndim > 1:
        # fil = np.tile(fil, (self.data.shape[1], 1)).T

        return fil

    @property
    def diff_mean_trend(self):
        """The trend (gradient) of predicted anomalies vs the trend of surrounding points, should be high"""
        if not 0 < self.num_anom < len(self.preds):
            return 0

        fil = self._get_mid_avg_filter()
        padding = self.window // 2

        # abs-trend avg
        diffs = []
        data = self.data if self.data.ndim > 1 else self.data[:, np.newaxis]
        for arr in data.T:
            grads = signal.savgol_filter(arr, self.window, 1, deriv=1, axis=0)
            grads = np.abs(grads)
            conv = np.abs(signal.convolve(grads, fil, mode="valid"))
            conv = np.pad(conv, (padding, padding), mode="edge")
            diffs.append(
                (
                    conv[self.preds == 1].mean(axis=0)
                    - conv[self.preds == 0].mean(axis=0)
                ).sum()
            )

        return np.mean(diffs)

    @property
    def diff_mid_avg(self):
        """The value of predicted anomalies vs the average of surrounding values, should be high"""
        if not 0 < self.num_anom < len(self.preds):
            return 0

        fil = self._get_mid_avg_filter()
        padding = self.window // 2

        # ~= midpoint minus avg. of sides
        diffs = []
        data = self.data if self.data.ndim > 1 else self.data[:, np.newaxis]
        for arr in data.T:
            conv = np.abs(signal.convolve(arr, fil, mode="valid"))
            conv = np.pad(conv, (padding, padding), mode="edge")
            diffs.append(
                conv[self.preds == 1].mean(axis=0)
                - conv[self.preds == 0].mean(axis=0).sum()
            )

        return np.mean(diffs)
