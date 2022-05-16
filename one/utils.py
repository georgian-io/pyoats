from os import listdir
from os.path import isfile, join

import numpy as np

def get_files_from_path(path: str):
    return sorted([f for f in listdir(path) if isfile(join(path, f))])

def array_safe_eq(a, b) -> bool:
    """Check if a and b are equal, even if they are numpy arrays"""
    if a is b:
        return True
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and (a == b).all()
    try:
        return a == b
    except TypeError:
        return NotImplemented

