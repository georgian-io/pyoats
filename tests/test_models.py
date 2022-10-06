import pytest

from oats.models import *
import tensorflow as tf


def has_gpu():
    return tf.test.is_gpu_available()


MODELS = [
    MovingAverageModel,
    QuantileModel,
    RegressionModel,
    LightGBMModel,
    RandomForestModel,
    ARIMAModel,
    NBEATSModel,
    TranADModel,
    IsolationForestModel,
    MatrixProfileModel,
    FluxEVModel,
    VAEModel,
]


@pytest.mark.models
@pytest.mark.parametrize("model", MODELS)
def test_model_univariate_1d(train_sv_1d, test_sv_1d, model):
    m = model()
    m.fit(train_sv_1d, epochs=1)
    res = m.get_scores(test_sv_1d)

    assert res.shape == (200,)


@pytest.mark.models
@pytest.mark.parametrize("model", MODELS)
def test_model_univariate_2d(train_sv_2d, test_sv_2d, model):
    m = model()
    m.fit(train_sv_2d, epochs=1)
    res = m.get_scores(test_sv_2d)

    assert res.shape == (200,)


@pytest.mark.models
@pytest.mark.parametrize("model", MODELS)
def test_model_multi_variate(train_mv, test_mv, model):
    m = model()
    m.fit(train_mv, epochs=1)
    res = m.get_scores(test_mv)

    assert res.shape == (200, 2)


GPU_MODELS = [NBEATSModel, TranADModel, MatrixProfileModel]


@pytest.mark.models
@pytest.mark.gpu
@pytest.mark.parametrize("model", GPU_MODELS)
def test_model_gpu(train_sv_1d, test_sv_1d, model):
    if not has_gpu():
        return
    m = model(use_gpu=True)
    m.fit(train_sv_1d, epochs=1)
    res = m.get_scores(test_sv_1d)

    assert res.shape == (200,)
