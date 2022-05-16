from typing import Protocol

import numpy as np


class Data(Protocol):
    pass

class DataReader(Protocol):
    def read(self, path: str) -> Data:
        raise NotImplementedError

