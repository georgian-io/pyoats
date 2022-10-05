from os import listdir
from os.path import isfile, join, isdir
from typing import Any

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def get_files_from_path(path: str):
    return sorted([f for f in listdir(path) if isfile(join(path, f))])


def get_dirs_from_path(path: str):
    return sorted([f for f in listdir(path) if isdir(join(path, f))])


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


def get_default_early_stopping():
    return EarlyStopping(monitor="val_loss", patience=5, min_delta=0.0001, mode="min")


def save_model_output(series: npt.NDArray[Any], model, fname: str):
    spec = str(model)
    np.savetxt(fname, series, header=spec)
