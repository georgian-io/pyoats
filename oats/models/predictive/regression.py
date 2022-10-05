"""
Regression
-----------------
"""
from darts import models

from oats.models._darts_simple import SimpleDartsModel


class RegressionModel(SimpleDartsModel):
    """Linear Regression Model

    Using linear regression as a predictor. Anomalies scores are deviations from predictions.

    Reference: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.regression.html
    """

    def __init__(
        self,
        window: int = 10,
        n_steps: int = 1,
        lags: int = 1,
        val_split: float = 0.2,
        **kwargs
    ):
        """
        initialization also accepts any parameters used by: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.regression.html

        Args:
            window (int, optional): rolling window size to feed into the predictor. Defaults to 10.
            n_steps (int, optional): number of steps to predict forward. Defaults to 1.
            lags (int, optional): number of lags. Defaults to 1.
            val_split (float, optional): proportion of data points reserved for validation; only used if using auto-tuning (not tested). Defaults to 0.
        """
        model_cls = models.RegressionModel

        super().__init__(model_cls, window, n_steps, lags, val_split, **kwargs)

    def hyperopt_model(self, *args):
        # overriding parent method as there's nothing to tune
        return
