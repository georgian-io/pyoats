import pytest
import numpy as np

from one.data.ucrdata import UcrDataReader, UcrData

@pytest.fixture
def data_reader():
    d = UcrDataReader()
    d.set_path("./tests/test_files/ucr/001_UCR_Anomaly_test_2_3_4.txt")
    return d

@pytest.fixture
def expected():
    series = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    labels = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    train_len = 2

    return UcrData(series, labels, train_len)


def test_read_existing(data_reader, expected):
    res = data_reader.read()

    assert res == expected

def test_read_nonexisting():
    d = UcrDataReader()
    d.set_path("./tests/test_files/nonsuch.txt")
    res = d.read()

    assert res is None

