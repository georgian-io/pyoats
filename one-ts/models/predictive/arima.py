import numpy as np

from statsmodels.tsa.arima.model import ARIMA

from one.models.base import Model


class ARIMAModel(Model):
    support_multivariate = False
    def __init__(self, p=1, d=1, q=1, **kwargs):
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


