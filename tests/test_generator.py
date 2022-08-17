import pytest

from one.generator import UnivariateWaveGenerator

@pytest.mark.generator
def test_univar_wave_generator():
    LEN = 1000
    gen = UnivariateWaveGenerator(LEN, train_ratio=0.2) 
    gen.point_global_outliers(0.01, 2, 50)
    gen.point_contextual_outliers(0.01, 2, 50)
    gen.collective_shapelet_outliers(0.01, 50)
    gen.collective_trend_outliers(0.01, 2, 50)
    gen.collective_seasonal_outliers(0.01, 2, 50)

    res = gen.get_dataset()
    train, test, label = res
    assert len(res) == 3
    assert len(train) == 0.2*LEN
    assert len(test) == 0.8*LEN
    assert label.sum() == pytest.approx(0.01 * LEN, abs=10)
