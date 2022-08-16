import pytest
import numpy as np

from one.scorer import Scorer

@pytest.fixture
def labels():
    return np.append(np.zeros(20), np.full(20, 1))

@pytest.fixture
def preds():
    pred = np.append(np.zeros(30), np.full(10, 1))
    pred[10] = 1
    return pred

@pytest.mark.scorer
def test_no_delay(labels, preds):
    tp = 20
    fp = 1
    tn = 19
    fn = 0

    scorer = Scorer()
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

    scorer = Scorer(15)
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

    scorer = Scorer(5)
    scorer.process(preds, labels)

    assert tp == scorer.tp
    assert fp == scorer.fp
    assert tn == scorer.tn
    assert fn == scorer.fn





    
