import pytest
import numpy as np

from one.data.ucrdata import UcrDataReader

def test_read_existing():
    d = UcrDataReader()
    d.set_path("./tests/test_files/ucr/001_UCR_Anomaly_test_2_3_4.txt")
    res = d.read()

    assert res == True

def test_read_nonexisting():
    d = UcrDataReader()
    d.set_path("./tests/test_files/nonsuch.txt")
    res = d.read()

    assert res == False

def test_series():
    d = UcrDataReader()
    d.set_path("./tests/test_files/ucr/001_UCR_Anomaly_test_2_3_4.txt")
    d.read()

    np.testing.assert_array_almost_equal(d.series, [2.0, 3.0, 4.0, 5.0, 6.0])

def test_labels():
    d = UcrDataReader()
    d.set_path("./tests/test_files/ucr/001_UCR_Anomaly_test_2_3_4.txt")
    d.read()

    np.testing.assert_array_almost_equal(d.labels, [0.0, 0.0, 1.0, 1.0, 0.0])

def test_train_len():
    d = UcrDataReader()
    d.set_path("./tests/test_files/ucr/001_UCR_Anomaly_test_2_3_4.txt")
    d.read()

    assert d.train_len == 2
