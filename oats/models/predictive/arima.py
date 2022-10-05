"""
ARIMA
-----------------
"""

import numpy as np

from statsmodels.tsa.arima.model import ARIMA

from oats.models._base import Model


class ARIMAModel(Model):
    """Autoregressive Intergrated Moving Average Model

    Implemented using statsmodels package. Multivariate scoring enabled by fitting and predicting each feature column.
    """

    def __init__(self, p=1, d=1, q=1, **kwargs):
        """
        Common parameters:
            - ARIMA(1,0,0) = first-order autoregressive model
            - ARIMA(0,1,0) = random walk
            - ARIMA(1,1,0) = differenced first-order autoregressive model
            - ARIMA(0,1,1) without constant = simple exponential smoothing
            - ARIMA(0,1,1) with constant = simple exponential smoothing with growth
            - ARIMA(0,2,1) or (0,2,2) without constant = linear exponential smoothing
            - ARIMA(1,1,2) with constant = damped-trend linear exponential smoothing`

        Args:
            p (int, optional): _description_. Defaults to 1.
            d (int, optional): _description_. Defaults to 1.
            q (int, optional): _description_. Defaults to 1.
        """
        self.order = (p, d, q)

    def fit(self, train, *args, **kwargs):
        if train.ndim > 1 and train.shape[1] > 1:
            self._models = self._pseudo_mv_train(train)
            return

        self.model = ARIMA(train, order=self.order)
        self.fitted = self.model.fit()

    def get_scores(self, data):
        if data.ndim > 1 and data.shape[1] > 1:
            return self._handle_multivariate(data, self._models)

        # test data has to be immediately after train
        scores = np.zeros(len(data))
        fitted = self.fitted
        for i in range(len(data)):
            forecast = fitted.forecast(1)
            scores[i] = forecast - data[i]

            fitted = fitted.append([data[i]], refit=False)

        return np.abs(scores)
