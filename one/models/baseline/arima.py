import numpy as np

from statsmodels.tsa.arima.model import ARIMA

from one.models.base import Model


class ARIMAModel(Model):
    def __init__(self, p, d, q):
        self.order = (p, d, q)

    def fit(self, train):
        self.model = ARIMA(train, order=self.order)
        self.fit = self.model.fit()

    def get_scores(self, data):
        # test data has to be immediately after train
        scores = np.zeros(len(data))
        fit = self.fit
        for i in range(len(data)):
            forecast = fit.forecast(1)
            scores[i] = forecast - data[i]

            fit = fit.append([data[i]], refit=False)

        return np.abs(scores)


