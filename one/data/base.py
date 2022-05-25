from typing import Protocol, Tuple, Any

import numpy as np
import numpy.typing as npt


class Data(Protocol):
    series: npt.NDArray[Any]
    labels: npt.NDArray[Any]
    train_len: int

    @property
    def train(self) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        raise NotImplementedError

    @property
    def test(self) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        raise NotImplementedError


class DataReader(Protocol):
    def __call__(self, path: str) -> Data:
        raise NotImplementedError

    def read(self) -> Data:
        raise NotImplementedError
