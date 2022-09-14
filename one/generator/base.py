from typing import Protocol, Tuple
import numpy.typing as npt


class Generator(Protocol):
    def get_dataset(
        self, *args, **kwargs
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        raise NotImplementedError
