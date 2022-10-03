from dataclasses import dataclass, astuple
from typing import Tuple, Any, Union

import numpy as np
import numpy.typing as npt

from oats._data.base import DataReader, Data
from oats._utils.utils import array_safe_eq


@dataclass(eq=False)
class UcrData(Data):
    series: npt.NDArray[Any]
    labels: npt.NDArray[Any]
    train_len: int
    file_name: str

    @property
    def train(self) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        # data, label
        tl = self.train_len
        return self.series[:tl], self.labels[:tl]

    @property
    def test(self) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        # data, label
        tl = self.train_len
        return self.series[tl:], self.labels[tl:]

    def get_test_with_window(
        self, window: int
    ) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        tl = self.train_len
        return self.series[tl - window :], self.labels[tl:]

    def __eq__(self, other) -> bool:
        if self is other:
            return True

        if self.__class__ is not other.__class__:
            return NotImplemented

        t1 = astuple(self)
        t2 = astuple(other)

        return all(array_safe_eq(a1, a2) for a1, a2 in zip(t1, t2))


class UcrDataReader(DataReader):
    def __init__(self):
        self.path: Union[str, None] = None

    def __call__(self, path: str) -> UcrData:
        self.set_path(path)
        return self.read()

    def set_path(self, path: str) -> None:
        self.path = path

    def read(self) -> UcrData:
        try:
            series = np.loadtxt(self.path)
        except FileNotFoundError:
            return None

        train_len = self._get_train_len()
        labels = self._get_labels(series)
        file_name = self.path.split("/")[-1]

        return UcrData(series, labels, train_len, file_name)

    def _get_labels(self, series: npt.NDArray[Any]) -> npt.NDArray[Any]:
        start_idx, end_idx = self._get_label_range()
        arr = np.zeros(series.size)
        arr[start_idx - 1 : end_idx] = 1.0

        return arr

    def _get_file_name(self) -> str:
        if self.path is None:
            return None

        return self.path.split("/")[-1].split(".")[0]

    def _get_train_len(self) -> int:
        if self.path is None:
            return None

        file_name = self._get_file_name()
        return int(file_name.split("_")[-3])

    def _get_label_range(self) -> Tuple[int, int]:
        if self.path is None:
            return None

        file_name = self._get_file_name()
        fn_split = file_name.split("_")
        return int(fn_split[-2]), int(fn_split[-1])
