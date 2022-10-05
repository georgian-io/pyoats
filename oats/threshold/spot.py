"""
Streaming Peaks-Over-Threshold (SPOT)
-----------------
"""
import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from statsmodels.robust.scale import huber
from scipy.stats import genpareto
from scipy.stats import norm

from oats.threshold._base import Threshold


class SPOTThreshold(Threshold):
    """Streaming version of POT
    Fit the tails of the data with Generalized Pareto Distribution (GPD).
    Find the threshold where `P(thres) < q`.
    Usual values for q is 1e-3 to 1e-6.

    `fit()` is required to compute the initial threshold for the first `n` points.

    Siffer, Alban and Fouque, Pierre-Alain and Termier, Alexandre and Largouet, Christine
    "Anomaly Detection in Streams with Extreme Value Theory"
    https://doi.org/10.1145/3097983.3098144
    """

    def __init__(
        self,
        q: float = 1e-4,
        level: float = 0.95,
        memory: int = 2000,
        support: float = 0,
        init_cutoff: float = 1,
        robust: bool = False,
        estimator: str = "MoM",
        **kwargs
    ):
        """
        Args:
            q (float, optional): q level such that `P(threshold) < q`. Defaults to 1e-4.
            level (float, optional): threshold to fit tail distribution. Defaults to 0.95.
            memory (int, optional): how many data points for the tails. Defaults to 2000.
            support (float, optional): used for method of moments estimation to prevent a bad fit (MoM is fast but unreliable); threshold is computed as the maximum of `GPD` and `support * HalfNormal`. Defaults to 0.
            init_cutoff (float, optional): used for method of moments estimation to prevent a bad fit (MoM is not robust against outliers); only bottom `init_cutoff` deciles are used to fit the GPD distribution. Defaults to 1.
            robust (bool, optional): (experimental) using robust estimates of `mean` and `variance` to fit GPD using MoM. Defaults to False.
            estimator (str, optional): estimator used to fit GPD distribution; accepts 'MoM' or 'MLE'. Defaults to "MoM".
        """
        self.support = support
        self.init_cutoff = init_cutoff
        self.robust = robust

        self.q = q
        self.level = level
        self.memory = memory

        # Rolling Data
        self.X = np.array([])

        # Peak Set
        self.Y = np.array([])

        # len(data set)
        self.n = 0

        # I -> indices of peaks in order to maintain the correct n_peak/n ratio
        self.I = np.array([])

        # thresholds
        self.percentile_thres = 0
        self.spot_thres = 0

        self.estimator = estimator

        self._thresholders = None

    @property
    def adjusted_n(self):
        return self.n - self.I[0]

    # TODO: typing
    #       data -> np.NDArray
    def fit(self, data):
        multivar = True if data.ndim > 1 and data.shape[1] > 1 else False
        if multivar:
            self._thresholders = self._pseudo_mv_fit(data)
            return

        for x in data:
            self._add_mem(x)

        self.spot_thres = self._spot_initialize(data)

    def _step(self, x_t) -> float:
        """
        Stepping + Updating SPOT
        """
        self._add_mem(x_t)
        new_p_thres = _GPDThreshold.get_tail_threshold(self.X, self.level)
        self.spot_thres += new_p_thres - self.percentile_thres
        self.percentile_thres = new_p_thres

        if self.X[-1] > self.spot_thres:
            """erasing anomalous data point from history"""
            self.X[-1] = 0

        elif self.X[-1] > self.percentile_thres:
            # add obs to peak set
            y_i = self.X[-1] - self.percentile_thres
            self._add_peak(y_i)

            sigma, gamma = _GPDThreshold.get_gpd_params(
                self.Y, robust=self.robust, estimator=self.estimator
            )
            self.spot_thres = max(
                _GPDThreshold.calc_spot_threshold(
                    self.percentile_thres,
                    sigma,
                    gamma,
                    self.adjusted_n,
                    self.Y.size,
                    self.q,
                ),
                _GPDThreshold.calc_half_normal_threshold(
                    self.percentile_thres,
                    self.Y.std(ddof=1),
                    self.q,
                    support=self.support,
                ),
            )

        return self.spot_thres

    def get_threshold(self, data):
        multivar = True if data.ndim > 1 and data.shape[1] > 1 else False
        if multivar:
            if not self._thresholders:
                self._thresholders = [self] * data.shape[1]
            return self._handle_multivariate(data, self._thresholders)

        ret = np.array([])
        for x in data:
            ret = np.append(ret, self._step(x))

        return ret

    @property
    def n_peaks(self):
        return self.Y.size

    def _spot_initialize(self, S):
        self.percentile_thres = _GPDThreshold.get_tail_threshold(S, self.level)

        Y = _GPDThreshold.get_peaks(S, self.percentile_thres)

        if self.init_cutoff < 1:
            cutoff = _GPDThreshold.get_tail_threshold(Y, self.init_cutoff)
            Y = Y[Y < cutoff]

        for y in Y:
            self._add_peak(y)

        sigma, gamma = _GPDThreshold.get_gpd_params(
            Y, robust=self.robust, estimator=self.estimator
        )

        n_y = len(Y)
        n = len(S)

        spot_thres = max(
            _GPDThreshold.calc_spot_threshold(
                self.percentile_thres, sigma, gamma, n, n_y, self.q
            ),
            _GPDThreshold.calc_half_normal_threshold(
                self.percentile_thres, Y.std(ddof=1), self.q, support=self.support
            ),
        )

        return spot_thres

    def _add_peak(self, y):
        if self.Y.size <= self.memory:
            self.Y = np.append(self.Y, y)
        else:
            self.Y = np.roll(self.Y, -1)
            self.Y[-1] = y

        if self.I.size <= self.memory + 1:
            self.I = np.append(self.I, self.n)
        else:
            self.I = np.roll(self.I, -1)
            self.I[-1] = self.n

    def _add_mem(self, x):
        if self.X.size <= self.memory:
            self.X = np.append(self.X, x)
        else:
            self.X = np.roll(self.X, -1)
            self.X[-1] = x
        self.n += 1


# SPOT based on MoM-Estimation Helper Class
class _GPDThreshold:
    @classmethod
    def get_tail_threshold(cls, data, level) -> float:
        return np.quantile(data, level)

    @classmethod
    def get_peaks(cls, data, threshold, n_sample_min=5):
        x = data.copy()
        x = x[x >= threshold]

        if len(x) < n_sample_min:
            threshold *= 0.95
            threshold = 0 if threshold < 1e-3 else threshold
            return cls.get_peaks(data, threshold, n_sample_min)

        return x - threshold

    @classmethod
    def get_gpd_params_mv(cls, mean, var):
        if estimator == "MoM":
            if var == 0:
                return 1, 1

            sigma = mean / 2 * (1 + mean**2 / var)
            gamma = 1 / 2 * (1 - (mean**2 / var))

        return sigma, gamma

    @classmethod
    def get_gpd_params(cls, peaks, robust=False, estimator="MoM"):
        y = peaks.copy()
        if estimator != "MoM":
            mu, sigma, gamma = genpareto.fit(y)
            return sigma, gamma

        if robust:
            try:
                huber.maxiter = 1000
                huber.tol = 5e-2
                mu, std = huber(y)
                var_y = std**2
            except ValueError:
                mu, var_y = y.mean(), y.var(ddof=1)
        else:
            mu, var_y = y.mean(), y.var(ddof=1)

        sigma = mu / 2 * (1 + mu**2 / var_y)
        gamma = 1 / 2 * (1 - (mu**2 / var_y))

        return sigma, gamma

    @classmethod
    def calc_spot_threshold(cls, initial_threshold, sigma, gamma, n, n_peaks, q):
        delta = sigma / gamma * ((q * n / n_peaks) ** (-gamma) - 1)
        new_thres = initial_threshold + delta

        return new_thres

    @classmethod
    def calc_half_normal_threshold(cls, initial_threshold, std_dev, q, support=1):
        return initial_threshold + std_dev * norm.ppf(((1 - q) + 1) / 2) * support
