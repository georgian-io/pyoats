from typing import Tuple
import numpy.typing as npt


class Generator():
    """Base class for Generators
    Because of complex behavior of generators. Method signatures are not strictly enforced.
    But must implement `get_dataset()` method that returns `(train, test, label)`
    """

    def get_dataset(
        self, *args, **kwargs
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        raise NotImplementedError
