import pytest

from one.models import *
import numpy as np


def get_sin(cycles, resolution):
    length = np.pi * 2 * cycles
    wave = np.sin(np.arange(0, length, length / resolution))
    return wave


@pytest.fixture
def train_sv_1d():
    wave = get_sin(2, 100)
    assert wave.ndim == 1
    return wave

@pytest.fixture
def test_sv_1d():
    wave = get_sin(4, 200)
    wave[99] = 10
    assert wave.ndim == 1
    return wave

@pytest.fixture
def train_sv_2d():
    wave = get_sin(2, 100)[:, np.newaxis]
    assert wave.ndim == 2
    return wave

@pytest.fixture
def test_sv_2d():
    wave = get_sin(4, 200)[:, np.newaxis]
    assert wave.ndim == 2
    return wave

@pytest.fixture
def train_mv():
    wave1 = get_sin(2, 100)
    wave2 = get_sin(4, 100)

    wave = np.vstack((wave1, wave2)).T
    assert wave.shape == (100, 2)
    return wave

@pytest.fixture
def test_mv():
    wave1 = get_sin(4, 200)
    wave2 = get_sin(8, 200)

    wave = np.vstack((wave1, wave2)).T
    assert wave.shape == (200, 2)
    return wave



MODELS = [MovingAverageModel, QuantileModel, RegressionModel, LightGBMModel, 
          RandomForestModel, ARIMAModel, NBEATSModel, TranADModel]
@pytest.mark.parametrize("model", MODELS)
def test_single_variate_1d(train_sv_1d, test_sv_1d, model):
    m = model()
    m.fit(train_sv_1d, epochs=1)
    res = m.get_scores(test_sv_1d)

    assert res.shape == (200, )

@pytest.mark.parametrize("model", MODELS)
def test_single_variate_2d(train_sv_2d, test_sv_2d, model):
    m = model()
    m.fit(train_sv_2d, epochs=1)
    res = m.get_scores(test_sv_2d)

    assert res.shape == (200, )

@pytest.mark.parametrize("model", MODELS)
def test_multi_variate(train_mv, test_mv, model):
    m = model()
    m.fit(train_mv, epochs=1)
    res = m.get_scores(test_mv)

    assert res.shape == (200, 2)





  



