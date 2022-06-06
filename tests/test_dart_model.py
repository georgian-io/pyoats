from unittest import mock

import pytest
import numpy as np

from one.models.predictive.darts_model import DartsModel



class MockModel(DartsModel):
    def __init__(
        self,
        window: int = 10,
        n_steps: int = 1,
        use_gpu: bool = False,
        val_split: float = 0.05,
    ):

        model = mock.Mock()

        super().__init__(model, window, n_steps, use_gpu, val_split)

    def _model_objective(self):
        return 0


@pytest.fixture
def mock_model(mocker):
    mocker.patch(
        "one.models.predictive.darts_model.get_default_early_stopping",
        return_value = True
    )

    mocker.patch(
        "one.models.predictive.darts_model.device_count",
        return_value = 1
    )

    mocker.patch(
        "one.models.predictive.darts_model.optuna",
    )

    return MockModel()


def test_init_model(mock_model):
    mock_model.model_cls.assert_called_with(10, 1, pl_trainer_kwargs={"callbacks": [True]})

def test_get_trainer_kwargs_cpu(mock_model):
    assert mock_model._get_trainer_kwargs() == {"callbacks": [True]}

def test_get_trainer_kwargs_gpu(mock_model):
    mock_model.use_gpu = True

    exp = {
        "accelerator": "gpu",
        "gpus": [0],
        "callbacks": [True]
    }

    assert mock_model._get_trainer_kwargs() == exp

def test_hyperopt_model(mock_model):
    mock_model.hyperopt_model(np.zeros(10), np.zeros(10), 2)
    mock_model.model_cls.assert_called_with(10, 1, pl_trainer_kwargs={"callbacks": [True]}, **mock_model.params)


def test_hyperopt_ws(mock_model):
    mock_model.model_cls.assert_called_with(10, 1, pl_trainer_kwargs={"callbacks": [True]})

def test_fit(mock_model):
    mock_model.fit(np.zeros(10))
    mock_model.model.fit.assert_called_once()


