import numpy as np

def set_initial_threshold(contamination: float, scores) -> float:
    return np.quantile(scores, contamination)

def get_peak_set(t: float, scores):
    x = scores.copy()
    x = x[x >= t]

    if len(x) == 0:
        t *= 0.95
        t = 0 if threshold < 1e-3 else threshold
        return get_peak_set(t, scores)

    return x - t

def get_gpd_param(peak_set):
    y = peak_set.copy()
    mu = y.mean()
    var_y = y.var(ddof=1)

    if var_y == 0: return 0, 1

    sigma = mu/2 * (1 + mu ** 2 / var_y)
    gamma = 1/2 * (1 - mu ** 2 / var_y)

    return sigma, gamma

def pot(scores, q, contamination = 0.98):
    t = set_initial_threshold(contamination, scores)
    y = get_peak_set(t, scores)
    n_y = len(y)
    n = len(scores)
    sigma, gamma = get_gpd_param(y)

    new_threshold = t + sigma/gamma * ((q*n/n_y)**(-gamma) - 1)
    return new_threshold

