import pytest

from oats.threshold import *
import numpy as np


THRES = [QuantileThreshold, POTThreshold, SPOTThreshold, JenksThreshold]
@pytest.mark.threshold
@pytest.mark.parametrize("thres", THRES)
def test_threshold_univariate_1d(test_sv_1d, thres):
    t = thres()
    t.fit(test_sv_1d)
    threshold = t.get_threshold(test_sv_1d)

    assert threshold.shape == (200, )

@pytest.mark.threshold
@pytest.mark.parametrize("thres", THRES)
def test_threshold_univariate_2d(test_sv_2d, thres):
    t = thres()
    t.fit(test_sv_2d)
    threshold = t.get_threshold(test_sv_2d)

    assert threshold.shape == (200, )

@pytest.mark.threshold
@pytest.mark.parametrize("thres", THRES)
def test_threshold_univariate_mv(test_mv, thres):
    t = thres()
    t.fit(test_mv)
    threshold = t.get_threshold(test_mv)

    assert threshold.shape == (200, 2)


@pytest.mark.threshold
@pytest.mark.parametrize("thres", THRES)
def test_threshold_univariate_1d_untrained(test_sv_1d, thres):
    t = thres()
    threshold = t.get_threshold(test_sv_1d)

    assert threshold.shape == (200, )

@pytest.mark.threshold
@pytest.mark.parametrize("thres", THRES)
def test_threshold_univariate_2d_untrained(test_sv_2d, thres):
    t = thres()
    threshold = t.get_threshold(test_sv_2d)

    assert threshold.shape == (200, )

@pytest.mark.threshold
@pytest.mark.parametrize("thres", THRES)
def test_threshold_univariate_mv_untrained(test_mv, thres):
    t = thres()
    threshold = t.get_threshold(test_mv)

    assert threshold.shape == (200, 2)