"""
Supervised Metrics
-----------------
"""

import numpy as np

from oats.scorer._base import Scorer


class SupervisedScorer(Scorer):
    """Scorer for traditional supervised metrics; only useful if actual anomalies are known"""

    def __init__(self, delay: int = None):
        """
        Args:
            delay (int, optional): For pattern anomalies, how much tolerance to give for prediction; e.g. delay of 10 means only those predictions in the first 10 time steps of a pattern anomalies are counted as success. None if no delay needed. Defaults to None.
        """
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

        self.delay = delay

    def process(self, preds, labels):
        preds = preds.copy()
        labels = labels.copy()

        ground_truth_ones = np.where(labels == 1)[0]
        pred_ones = np.where(preds == 1)[0]
        ranges = self._consecutive(ground_truth_ones)

        tp, fp, tn, fn = 0, 0, 0, 0

        for idx, r in enumerate(ranges):
            intersect = np.intersect1d(r, pred_ones, assume_unique=True)
            # if alert delay more than 100 timesteps, count that as bad!

            if intersect.size != 0:
                cond = (
                    intersect[0] < r[0] + self.delay if self.delay is not None else True
                )
                if cond:
                    tp += r.size
                else:
                    fn += r.size

                preds[intersect] = 0
                pred_ones = np.where(preds == 1)[0]
            else:
                fn += r.size

        fp += pred_ones.size
        tn += preds.size - tp - fp - fn

        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def _consecutive(self, data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)

    @property
    def tpr(self):
        """True Positive Rate"""

        return self.tp / (self.fn + self.tp)

    @property
    def fpr(self):
        """False Positive Rate"""
        return self.fp / (self.tn + self.fp)

    @property
    def tnr(self):
        """True Negative Rate"""
        if self.tn + self.fp == 0:
            return 0
        return self.tn / (self.tn + self.fp)

    @property
    def fnr(self):
        """False Negative Rate"""
        if self.fn + self.tp == 0:
            return 0
        return self.fn / (self.fn + self.tp)

    @property
    def precision(self):
        """Precision"""
        if self.tp + self.fp == 0:
            return 0
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self):
        """Recall"""
        if self.tp + self.fn == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    @property
    def f1(self):
        """F-1 Score"""
        if self.recall + self.precision == 0:
            return 0
        return (2 * self.precision * self.recall) / (self.precision + self.recall)

    def __str__(self):
        return f"{self.tp}, {self.fp}, {self.tn}, {self.fn}, {self.tpr}, {self.fpr}, {self.tnr}, {self.fnr}, {self.precision}, {self.recall}, {self.f1}"
