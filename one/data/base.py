from typing import Protocol

import numpy as np

class DataReader(Protocol):
    def read(self, path: str) -> bool:
        raise NotImplementedError

