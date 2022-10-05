"""
Spectral Residual
-------------------
"""


# Copyright (c) 2019 Takahiro Yoshinaga

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from oats.preprocessor._base import Preprocessor


def _series_filter(values, kernel_size=3):
    """
    Filter a time series. Practically, calculated mean value inside kernel size.
    As math formula, see https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html.

    Args:
        values:
        kernel_size:

    Returns:
        The list of filtered average
    """
    filter_values = np.cumsum(values, dtype=float)

    filter_values[kernel_size:] = (
        filter_values[kernel_size:] - filter_values[:-kernel_size]
    )
    filter_values[kernel_size:] = filter_values[kernel_size:] / kernel_size

    for i in range(1, kernel_size):
        filter_values[i] /= i + 1

    return filter_values


def _extrapolate_next(values):
    """
    Extrapolates the next value by sum up the slope of the last value with previous values.
    :param values: a list or numpy array of time-series
    :return: the next value of time-series
    """

    last_value = values[-1]
    slope = [(last_value - v) / i for (i, v) in enumerate(values[::-1])]
    slope[0] = 0
    next_values = last_value + np.cumsum(slope)

    return next_values


def _merge_series(values, extend_num=5, forward=5):

    next_value = _extrapolate_next(values)[forward]
    extension = [next_value] * extend_num

    if isinstance(values, list):
        marge_values = values + extension
    else:
        marge_values = np.append(values, extension)
    return marge_values


class SpectralResidual(Preprocessor):
    """Generates a saliency map via spectral residual.

    Inspired from:
        Hansheng Ren, Bixiong Xu, Yujing Wang, Chao Yi, Congrui Huang, Xiaoyu Kou, Tony Xing, Mao Yang, Jie Tong, Qi Zhang.
        "Time-Series Anomaly Detection Service at Microsoft." arXiv preprint arXiv:1906.03821 (2019).

    Implementation: https://github.com/y-bar/ml-based-anomaly-detection
    """

    def __init__(self, amp_window_size=16, series_window_size=16, score_window_size=32):
        self.amp_window_size = amp_window_size
        self.series_window_size = series_window_size
        self.score_window_size = score_window_size

    def _transform_silency_map(self, values):
        """
        Transform a time-series into spectral residual, a method adopted from computer vision.
        For example, See https://github.com/uoip/SpectralResidualSaliency.

        Args:
            values: a list or numpy array of float values.

        Returns:
            Silency map and spectral residual
        """

        freq = np.fft.fft(values)
        mag = np.sqrt(freq.real**2 + freq.imag**2)
        spectral_residual = np.exp(
            np.log(mag) - _series_filter(np.log(mag), self.amp_window_size)
        )

        freq.real = freq.real * spectral_residual / mag
        freq.imag = freq.imag * spectral_residual / mag

        silency_map = np.fft.ifft(freq)
        return silency_map

    def _transform_spectral_residual(self, values):
        silency_map = self._transform_silency_map(values)
        spectral_residual = np.sqrt(silency_map.real**2 + silency_map.imag**2)
        return spectral_residual

    def transform(self, values: ArrayLike, type: str = "avg") -> NDArray:
        """
        Transform series using Spectral Residual

        Args:
            values: timeseries
            type: filter type in `["avg", "abs", "chisq"]`

        Returns:

        """
        multivar = True if values.ndim > 1 and values.shape[1] > 1 else False
        if multivar:
            return self._handle_multivariate(values, [self] * values.shape[1])

        extended_series = _merge_series(
            values, self.series_window_size, self.series_window_size
        )
        mag = self._transform_spectral_residual(extended_series)[: len(values)]

        if type == "avg":
            ave_filter = _series_filter(mag, self.score_window_size)
            score = (mag - ave_filter) / ave_filter
        elif type == "abs":
            ave_filter = _series_filter(mag, self.score_window_size)
            score = np.abs(mag - ave_filter) / ave_filter
        elif type == "chisq":
            score = stats.chi2.cdf((mag - np.mean(mag)) ** 2 / np.var(mag), df=1)
        else:
            raise ValueError("No type!")
        return score

    def fit(self, *arg, **kwargs) -> None:
        return
