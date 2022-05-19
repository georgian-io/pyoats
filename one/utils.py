from os import listdir
from os.path import isfile, join
from typing import Any

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from one.constants import *

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

def graph_data(series: npt.NDArray[Any],
               labels: npt.NDArray[Any],
               train_len: int):

    # create fig and axes
    fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={"hspace":HSPACE, "height_ratios":HRATIO})

    # plot data
    ax.plot(series)

    # adding annotation
    ax.axvline(x=train_len-1, linestyle="--", c="red", lw=5, zorder=100)
    ax.annotate("Training Data Ends",
                (train_len * 1.01,ax.get_ylim()[1]),
                fontsize="x-large")

    # styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # plot label
    ax2.imshow(labels[np.newaxis,:], cmap=LABELS_CMAP, aspect="auto", interpolation="none", vmin=0, vmax=1)
    ax2.set_yticks([])


    return fig


def get_default_early_stopping():
    return EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=0.0001,
        mode='min'
    )



