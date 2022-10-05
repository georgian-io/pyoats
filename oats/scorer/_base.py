from typing import Protocol


class Scorer(Protocol):
    """Base class for Scorers.
    Scorer computes the result of a predictor; can be either unsupervised or supervised.

    Under supervised setting, predicted labels are compared against actual labels to compute traditional metrics such as `f1`.
    Under unsupervised setting, anomaly scores and/or predicted labels are usually used in conjunction with original data to compute metrics.

    Example:
        # get anomaly scores
        anom_score = model.get_scores()

        # get threshold
        thresholder = QuantileThreshold()
        threshold = thresholder.get_threshold(anom_score, 0.95)

        preds = anom_score > threshold

        # Scoring output
        scorer = SupervisedScorer()
        scorer.process(preds, labels)

        print(f"F1: {scorer.f1}")
        print(f"Precision: {scorer.precision}")
        print(f"Recall: {scorer.recall}")
    """

    def process(self, *args, **kwargs):
        raise NotImplementedError
