import numpy as np



class POT:
    @classmethod
    def _set_initial_threshold(cls, contamination: float, scores) -> float:
        return np.quantile(scores, contamination)

    @classmethod
    def _get_peak_set(cls, t: float, scores):
        x = scores.copy()
        x = x[x >= t]

        if len(x) == 0:
            t *= 0.95
            t = 0 if threshold < 1e-3 else threshold
            return get_peak_set(t, scores)

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

    def get_thres(self, scores, q, contamination = 0.98):
        t = cls._set_initial_threshold(contamination, scores)
        y = cls._get_peak_set(t, scores)
        n_y = len(y)
        n = len(scores)
        sigma, gamma = cls.get_gpd_param(y)

        new_threshold = t + sigma/gamma * ((q*n/n_y)**(-gamma) - 1)
        return new_threshold