import pytest
import numpy as np

from one.scorer import *

@pytest.fixture
def labels():
    return np.append(np.zeros(20), np.full(20, 1))

@pytest.fixture
def preds():
    pred = np.append(np.zeros(30), np.full(10, 1))
    pred[10] = 1
    return pred

@pytest.fixture
def data():
    return np.append(np.arange(20), np.arange(19, -1, -1))

@pytest.mark.scorer
def test_no_delay(labels, preds):
    tp = 20
    fp = 1
    tn = 19
    fn = 0

    scorer = SupervisedScorer()
    scorer.process(preds, labels)

    assert tp == scorer.tp
    assert fp == scorer.fp
    assert tn == scorer.tn
    assert fn == scorer.fn

@pytest.mark.scorer
def test_delay_within(labels, preds):
    tp = 20
    fp = 1
    tn = 19
    fn = 0

    scorer = SupervisedScorer(15)
    scorer.process(preds, labels)

    assert tp == scorer.tp
    assert fp == scorer.fp
    assert tn == scorer.tn
    assert fn == scorer.fn



@pytest.mark.scorer
def test_delay_without(labels, preds):
    tp = 0
    fp = 1
    tn = 19
    fn = 20

    scorer = SupervisedScorer(5)
    scorer.process(preds, labels)

    assert tp == scorer.tp
    assert fp == scorer.fp
    assert tn == scorer.tn
    assert fn == scorer.fn


@pytest.mark.scorer
def test_qualitative(data, preds):
    scorer = QualitativeMetrics()
    scorer.process(data, preds)

    assert scorer.num_anom == 11
    assert scorer.pct_anom == 11/40
    assert scorer.avg_anom_dist_from_mean == np.abs(np.array([9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]) - np.mean(data)).mean()
    assert scorer.avg_cycles_delta_between_anom == np.array([20, 1, 1, 1, 1, 1, 1, 1, 1, 1]).mean()
    assert scorer.max_range_non_anom == 19
    assert scorer.diff_mean_trend == pytest.approx(0, abs=0.5)
    assert scorer.diff_mid_avg == pytest.approx(0, abs=0.5)


@pytest.mark.scorer
def test_qualitative_2d_sv(data, preds):
    data = data[:, np.newaxis]
    scorer = QualitativeMetrics()
    scorer.process(data, preds)

    assert scorer.num_anom == 11
    assert scorer.pct_anom == 11/40
    assert scorer.avg_anom_dist_from_mean == np.abs(np.array([9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]) - np.mean(data)).mean()
    assert scorer.avg_cycles_delta_between_anom == np.array([20, 1, 1, 1, 1, 1, 1, 1, 1, 1]).mean()
    assert scorer.max_range_non_anom == 19
    assert scorer.diff_mean_trend == pytest.approx(0, abs=0.5)
    assert scorer.diff_mid_avg == pytest.approx(0, abs=0.5)


@pytest.mark.scorer
def test_qualitative_mv(data, preds):
    data = np.vstack((data, data)).T
    scorer = QualitativeMetrics()
    scorer.process(data, preds)

    assert scorer.num_anom == 11
    assert scorer.pct_anom == 11/40
    assert scorer.avg_anom_dist_from_mean == np.abs(np.array([9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]) - np.mean(data)).mean()
    assert scorer.avg_cycles_delta_between_anom == np.array([20, 1, 1, 1, 1, 1, 1, 1, 1, 1]).mean()
    assert scorer.max_range_non_anom == 19
    assert scorer.diff_mean_trend == pytest.approx(0, abs=0.5)
    assert scorer.diff_mid_avg == pytest.approx(0, abs=0.5)

@pytest.mark.scorer
def test_qualitative_non_matching_data_preds_shape(data, preds):
    data = np.append(data, data)

    scorer = QualitativeMetrics()

    with pytest.raises(ValueError):
        scorer.process(data, preds)

@pytest.mark.scorer
def test_qualitative_non_matching_data_shape(data, preds):
    data1 = data
    data2 = np.vstack((data, data)).T

    scorer = QualitativeMetrics()
    scorer.process(data1, preds)

    with pytest.raises(ValueError):
        scorer.process(data2, preds)



    
