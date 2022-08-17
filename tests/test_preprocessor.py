import pytest

from one.preprocessor import *
import numpy as np
import tensorflow as tf

PROCESSORS = [SpectralResidual]
@pytest.mark.preprocessors
@pytest.mark.parametrize("proc", PROCESSORS)
def test_processor_univariate_1d(test_sv_1d, proc):
    p = proc()
    res = p.transform(test_sv_1d)
    assert res.shape == (200, )

@pytest.mark.preprocessors
@pytest.mark.parametrize("proc", PROCESSORS)
def test_processor_univariate_2d(test_sv_2d, proc):
    p = proc()
    res = p.transform(test_sv_2d)
    assert res.shape == (200, )

@pytest.mark.preprocessors
@pytest.mark.parametrize("proc", PROCESSORS)
def test_processor_multivariate(test_mv, proc):
    p = proc()
    res = p.transform(test_mv)
    assert res.shape == (200, 2)