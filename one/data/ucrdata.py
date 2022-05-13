from typing import Tuple, Any

import numpy as np
import numpy.typing as npt

from one.data.base import DataReader

class UcrDataReader(DataReader):
    def __init__(self):
        self.path: str = None
        self.series: npt.NDArray[Any] = None
        self.labels: npt.NDArray[Any] = None
        self.train_len: int = None

        return

    def set_path(self, path:str) -> None:
        self.path = path

        return

    def read(self) -> bool:
        try:
            self.series = np.loadtxt(self.path)
        except FileNotFoundError:
            return False

        self.train_len = self._get_train_len()
        self.labels = self._get_labels()

        return True

    def _get_labels(self) -> npt.NDArray[Any]:
        if self.series is None: return None

        start_idx, end_idx = self._get_label_range()
        arr = np.zeros(self.series.size)
        arr[start_idx-1: end_idx] = 1.

        return arr

    def _get_file_name(self) -> str:
        if self.path is None: return None

        return self.path.split("/")[-1].split(".")[0]


    def _get_train_len(self) -> int:
        if self.path is None: return None

        file_name = self._get_file_name()
        return int(file_name.split("_")[-3])


    def _get_label_range(self) -> Tuple[int, int]:
        if self.path is None: return None

        file_name = self._get_file_name()
        fn_split = file_name.split("_")
        return int(fn_split[-2]), int(fn_split[-1])
