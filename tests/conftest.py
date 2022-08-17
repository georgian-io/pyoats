import pytest
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
    wave[99] = 10
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
    wave[99] = 10, 10
    assert wave.shape == (200, 2)
    return wave

