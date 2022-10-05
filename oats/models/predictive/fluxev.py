"""
FluxEV
-----------------
"""

import time

from oats.models._base import Model
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import norm, genpareto


class _FluxEV2000:
    def __init__(
        self,
        window,
        q,
        level,
        alpha=0.7,
        method="all",
        p=None,
        memory=2000,
        bw_reset_every=2000,
        support=0,
        init_cutoff=1,
        mle=False,
        **kwargs,
    ):
        if method.lower() not in ["all", "above", "below"]:
            raise ValueError(
                f"method parameter expexts ['all', 'above', 'below'] but passed in {method}"
            )
        self.bw_reset_every = bw_reset_every
        self.support = support
        self.init_cutoff = init_cutoff

        self.s = window
        self.q = q
        self.level = level
        self.p = 1 if p is None else p
        self.memory = memory

        # EWMA Decay Factor
        # alpha -> 0 => normal EMA
        # alpha -> 1 => EWMA -> X[-1]
        self.alpha = alpha

        # Method: Literal["all", "above", "below"]
        # all    => admit all points
        # above  => admit only those above moving avg
        # below  => admit only those below moving average
        self.method = method

        # Raw Data
        self.X = np.zeros(window)

        # Residual after EWMA
        self.E = np.zeros(window)

        # Residual after First Smoothing
        self.F = np.zeros(p)

        # Peak Set
        self.Y = np.array([])

        # F Memory
        self.F_mem = np.array([])

        # S -> Residual after second smoothing
        self.S = np.zeros(1)

        # len(data set)
        self.n = 0

        # I -> indices of peaks in order to maintain the correct n_peak/n ratio
        self.I = np.array([0])

        # thresholds
        self.percentile_thres = 0
        self.spot_thres = 0

        # timer
        self.perf_counter = 0

        # Use MLE Estimation (more stable)
        self.mle = mle

    @property
    def adjusted_n(self):
        return self.n - self.I[0]

    # TODO: typing
    #       data -> np.NDArray
    def load_initial(self, data):
        if data.size <= 2 * self.s:
            raise ValueError(
                f"Expects initial series of length > max(2*s, p) == {max(2*self.s, self.p)}, passed in {data.size}"
            )
        t = time.perf_counter()

        self.X = data[self.s : 2 * self.s]

        s_window = sliding_window_view(data[: 2 * self.s], self.s)[:-1]
        self.E = data[self.s : 2 * self.s] - EWMA(s_window.T, self.alpha)

        Q = np.array([])
        for i in data[2 * self.s :]:
            # initialize SPOT
            self._step_train(i)
            Q = np.append(Q, self.F[-1])

        self.spot_thres = self._spot_initialize(Q)
        self.perf_counter += time.perf_counter() - t

    # TODO: typing
    #       data -> np.float / float
    def _step_train(self, x_t) -> int:
        """
        Stepping without updating SPOT
        """
        # fluctuation removal
        E_i = x_t - EWMA(self.X, self.alpha)

        if self.method == "above":
            E_i = np.maximum(0.0, E_i)
        if self.method == "below":
            E_i = np.minimum(0, E_i)

        # first smoothing
        E_new = np.append(self.E, E_i)
        E_old = self.E.copy()
        F_i = (np.std(E_new, ddof=1) - np.std(E_old, ddof=1)).clip(min=0)

        self._add_mem(F_i)

        # second smoothing
        m = np.max(self.F)
        self.S = np.where(F_i - m > 0, np.array([F_i]), np.array([0]))

        # smoothing calculations done; advance one step
        self._array_step(x_t, E_i, F_i)

    def step_predict(self, x_t) -> int:
        """
        Stepping + Updating SPOT
        """
        t = time.perf_counter()
        if self.n % self.bw_reset_every == 0:
            print(f"z_t: {self.percentile_thres} | z_q: {self.spot_thres}")
            self.bw = _SPOTMoM.fit_kde_bw(self.F_mem)

        self._step_train(x_t)
        new_p_thres = _SPOTMoM.get_tail_threshold(
            self.F_mem, self.level, samples=100, bw=self.bw
        )
        self.spot_thres += new_p_thres - self.percentile_thres
        self.percentile_thres = new_p_thres

        out = np.int32(0)
        if self.S[-1] > self.spot_thres:
            """erasing anomalous data point from history"""
            out = np.int32(1)
            self.F_mem[-1] = 0

        elif self.F[-1] > self.percentile_thres and self.F[-1] < self.spot_thres:
            # add obs to peak set
            y_i = self.F[-1] - self.percentile_thres
            self._add_peak(y_i)

            """ vanilla calc
            sigma, gamma = SPOTMoM.get_gpd_params(self.Y)
            self.spot_thres = SPOTMoM.calc_spot_threshold(self.percentile_thres, sigma, gamma, self.n, self.Y.size, self.q)
            """

            sigma, gamma = _SPOTMoM.get_gpd_params(self.Y, mle=self.mle)
            self.spot_thres = max(
                _SPOTMoM.calc_spot_threshold(
                    self.percentile_thres,
                    sigma,
                    gamma,
                    self.adjusted_n,
                    self.Y.size,
                    self.q,
                ),
                _SPOTMoM.calc_half_normal_threshold(
                    self.percentile_thres,
                    self.Y.std(ddof=1),
                    self.q,
                    support=self.support,
                ),
            )

        elif self.F[-1] > self.spot_thres:
            self.F_mem[-1] = 0

        self.perf_counter += time.perf_counter() - t
        return out

    def _array_step(self, X_i, E_i, F_i):
        self.X = np.roll(self.X, -1)
        self.X[-1] = X_i

        self.E = np.roll(self.E, -1)
        self.E[-1] = E_i

        self.F = np.roll(self.F, -1)
        self.F[-1] = F_i

        self.n += 1

    @property
    def n_peaks(self):
        return self.Y.size

    def _spot_initialize(self, S):
        # print(f"Q_len: {len(S)} | Q_non_zeros: {np.count_nonzero(S)} | Q_max: {np.max(S)}")
        # print(S[S>0])
        self.bw = _SPOTMoM.fit_kde_bw(S)
        self.percentile_thres = _SPOTMoM.get_tail_threshold(S, self.level, bw=self.bw)

        # print(f"thres: {self.percentile_thres}")
        # print(f"n_above: {((S-self.percentile_thres) > 0).sum()}")

        Y = _SPOTMoM.get_peaks(S, self.percentile_thres)
        I = _SPOTMoM.get_peaks_idx(S, self.percentile_thres)

        if self.init_cutoff < 1:
            bw = _SPOTMoM.fit_kde_bw(Y)
            cutoff = _SPOTMoM.get_tail_threshold(Y, self.init_cutoff, bw=bw)
            Y = Y[Y < cutoff]

        for y, i in zip(Y, I):
            self._add_peak(y, i)

        sigma, gamma = _SPOTMoM.get_gpd_params(Y, mle=self.mle)

        n_y = len(Y)
        n = len(S)

        spot_thres = max(
            _SPOTMoM.calc_spot_threshold(
                self.percentile_thres, sigma, gamma, n, n_y, self.q
            ),
            _SPOTMoM.calc_half_normal_threshold(
                self.percentile_thres, Y.std(ddof=1), self.q, support=self.support
            ),
        )

        return spot_thres

    def _add_peak(self, y, index=None):
        if self.Y.size <= self.memory:
            self.Y = np.append(self.Y, y)
        else:
            self.Y = np.roll(self.Y, -1)
            self.Y[-1] = y

        index = self.n if index is None else index
        if self.I.size <= self.memory + 1:
            self.I = np.append(self.I, index)
        else:
            self.I = np.roll(self.I, -1)
            self.I[-1] = index

    def _add_mem(self, f):
        if self.F_mem.size <= self.memory:
            self.F_mem = np.append(self.F_mem, f)
        else:
            self.F_mem = np.roll(self.F_mem, -1)
            self.F_mem[-1] = f


def EWMA(x, alpha, flip_x=True):
    if flip_x:
        x = np.flip(x)

    s = len(x)
    a = np.full(s, 1 - alpha)
    power = np.arange(0, s)
    weights = np.power(a, power)

    res = np.dot(weights.T, x) / np.sum(weights)
    return np.flip(res)


class _SPOTMoM:
    @classmethod
    def fit_kde_bw(cls, data, kernel="gaussian"):
        params = {"bandwidth": np.logspace(-5, 10, 100)}
        grid = GridSearchCV(KernelDensity(rtol=0.01), params, n_jobs=-1)
        grid.fit(data[:, np.newaxis])
        return grid.best_params_["bandwidth"]

    @classmethod
    def get_tail_threshold(
        cls, data, level, samples=1000, bw=0.5, kernel="gaussian", rtol=0.02
    ) -> float:
        # via KDE
        data = np.append(data, 1e-3)  # added for stability in case data is all zeros
        x = np.linspace(data.min() + 1e-3, data.max(), samples)

        kde = KernelDensity(kernel=kernel, rtol=rtol, bandwidth=bw).fit(
            data[:, np.newaxis]
        )
        pdf = np.exp(kde.score_samples(x[:, np.newaxis]))

        cdf = np.cumsum(pdf)
        cdf /= cdf.max()

        if len(np.where(cdf <= level)[0]) == 0:
            return cls.get_tail_threshold(data, level, samples * 5, bw, kernel)

        idx = np.where(cdf <= level)[0][-1]

        return x[idx]

    @classmethod
    def get_peaks(cls, data, threshold, n_sample_min=5):
        threshold = cls._adjust_threshold(data, threshold, n_sample_min)
        return data[data >= threshold] - threshold

    @classmethod
    def get_peaks_idx(cls, data, threshold, n_sample_min=5):
        threshold = cls._adjust_threshold(data, threshold, n_sample_min)
        return np.where(data >= threshold)[0]

    @classmethod
    def _adjust_threshold(cls, data, threshold, n_sample_min):
        x = data.copy()
        x = x[x >= threshold]

        if len(x) < n_sample_min:
            threshold *= 0.95
            threshold = 0 if threshold < 1e-3 else threshold
            return cls._adjust_threshold(data, threshold, n_sample_min)

        return threshold

    @classmethod
    def get_gpd_params_mv(cls, mean, var):
        if var == 0:
            return 1, 1

        sigma = mean / 2 * (1 + mean**2 / var)
        gamma = 1 / 2 * (1 - (mean**2 / var))

        return sigma, gamma

    @classmethod
    def get_gpd_params(cls, peaks, robust=False, mle=False):
        y = peaks.copy()

        if mle:
            mu, sigma, gamma = genpareto.fit(y)
            return sigma, gamma

        if robust:
            huber.maxiter = 100
            huber.tol = 5e-2
            mu, std = huber(y)
            var_y = std**2
        else:
            mu = y.mean()
            var_y = y.var(ddof=1)

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


class FluxEVModel(Model):
    """FluxEV

    Implementation from the paper with QoL improvements for production.

    Li, Jia and Di, Shimin and Shen, Yanyan and Chen, Lei
    "FluxEV: A Fast and Effective Unsupervised Framework for Time-Series Anomaly Detection"
    https://doi.org/10.1145/3437963.3441823
    """

    def __init__(
        self, window: int = 10, window_smoothing=None, q=1e-4, level=0.95, **kwargs
    ):
        """
        Args:
            window (int, optional): main window length. Defaults to 10.
            window_smoothing (int, optional): window length for smoothig operation, if None then same as `window`. Defaults to None.
            q (float, optional): q level such that `P(threshold) < q`. Defaults to 1e-4.
            level (float, optional): threshold to fit tail distribution. Defaults to 0.95.
            memory (int, optional): how many data points for the tails. Defaults to 2000.
            bw_reset_every (int, optional): used for method of moments estimation to prevent a bad fit (MoM is not robust against outliers); use distribution from kde estimation to compute tail level; this parameter dictates how often bandwidth is refitted. Defaults to 2000.
            support (float, optional): used for method of moments estimation to prevent a bad fit (MoM is fast but unreliable); threshold is computed as the maximum of `GPD` and `support * HalfNormal`. Defaults to 0.
            init_cutoff (float, optional): used for method of moments estimation to prevent a bad fit (MoM is not robust against outliers); only bottom `init_cutoff` deciles are used to fit the GPD distribution. Defaults to 1.
            mle (bool, optional): whether to use MLE for GPD parameter estimation; setting True will result in a slow down in performance. Default to False.

        """
        self.window = window
        self.window_smoothing = window if window_smoothing is None else window_smoothing
        self.q = q
        self.level = level

        self.params = kwargs

    def fit(self, data, **kwargs):
        if data.ndim > 1 and data.shape[1] == 1:
            data = data.flatten()

        if data.ndim > 1 and data.shape[1] > 1:
            self._models = self._pseudo_mv_train(data)
            return

        self.model = _FluxEV2000(
            self.window,
            q=self.q,
            level=self.level,
            p=self.window_smoothing,
            **self.params,
        )
        self.model.load_initial(data)

    def get_scores(self, data):
        multivar = True if data.ndim > 1 and data.shape[1] > 1 else False
        if multivar:
            return self._handle_multivariate(data, self._models)

        if data.ndim > 1 and data.shape[1] == 1:
            data = data.flatten()

        anoms = np.zeros(len(data))
        for idx, value in enumerate(data):
            anoms[idx] = self.model.step_predict(value)

        return anoms
