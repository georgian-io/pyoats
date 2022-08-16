import numpy as np
import scipy.signal as signal

class QualitativeMetrics:
    def __init__(self, window=10):
        self.data = np.array([])
        self.preds = np.array([])

        self.window = window

    def process(self, data, preds):
        self.data = np.append(self.data, data)
        self.preds = np.append(self.preds, preds)

    @property
    def num_anomalies(self):
        return self.preds.sum()

    @property
    def percent_anomalies(self):
        return self.num_anomalies/len(self.preds)

    @property
    def _pred_anomalies(self):
        return self.data[self.preds == 1]

    @property
    def _pred_non_anomalies(self):
        return self.data[self.preds == 0]

    @property
    def avg_anom_dist_from_mean(self):
        return np.linalg.norm(self._pred_anomalies - self.data.mean(axis=0), axis=-1).mean()

    @property
    def avg_cycles_delta_between_anom(self):
        if self.num_anomalies not in (0, 1, len(self.data)):
            return 0
        return np.diff(np.where(self.preds==1)[0]).mean()

    @property
    def max_range_non_anom(self):
        if self.num_anomalies not in (0, len(self.data)):
            return 1e5
        return (np.abs(self.pred_non_anomalies.max() - self.pred_non_anomalies.min())).mean()

    def _get_mid_avg_filter(self):
        # make sure window is odd
        if self.window % 2 == 0: self.window += 1
        padding = self.window//2
        
        # local difference filter
        fil = self.window
        fil = np.full((self.window),-1/(self.window-1))
        fil[padding] = 1
        if self.data.ndim > 1:
            fil = np.tile(fil, (self.data.shape[1], 1)).T
 
        return fil 

    @property
    def diff_mean_trend(self):
        if not 0 < self.num_anomalies < len(self.preds): 
            return 0

        fil = self._get_mid_avg_filter()
        padding = self.window//2

        # abs-trend avg
        grads = signal.savgol_filter(self.data, window, 1, deriv=1, axis=0)
        grads = np.abs(grads)
        conv = np.abs(signal.convolve(grads, fil, mode="valid"))
        conv = np.pad(conv, (padding, padding), mode="edge")

        return (conv[self.preds==1].mean(axis=0) - conv[self.preds==0].mean(axis=0)).sum() 

    @property
    def dif_mid_avg(self):
        if not 0 < self.num_anomalies < preds.size: return 0

        fil = self._get_mid_avg_filter()
        padding = self.window//2

        # ~= midpoint minus avg. of sides
        conv = np.abs(signal.convolve(self.data, fil, mode="valid"))
        conv = np.pad(conv, (padding, padding), mode="edge")
        return (conv[self.preds==1].mean(axis=0) - conv[self.preds==0].mean(axis=0)).sum()

