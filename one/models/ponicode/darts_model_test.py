from one.models import darts_model
import pytest


class Test_Dartsmodel_Get_scores:
    @pytest.fixture()
    def dartsmodel(self):
        return darts_model.DartsModel()

    def test_get_scores_1(self, dartsmodel):
        result = dartsmodel.get_scores(
            ["Anas", "Anas", "Edmond", "Michael", "George", "Edmond"]
        )

    def test_get_scores_2(self, dartsmodel):
        result = dartsmodel.get_scores(
            ["Edmond", "Michael", "Michael", "George", "George", "Edmond"]
        )

    def test_get_scores_3(self, dartsmodel):
        result = dartsmodel.get_scores(
            ["Edmond", "Michael", "Edmond", "Anas", "Pierre Edouard", "Pierre Edouard"]
        )

    def test_get_scores_4(self, dartsmodel):
        result = dartsmodel.get_scores(
            [
                "Anas",
                "Jean-Philippe",
                "Pierre Edouard",
                "Michael",
                "Anas",
                "Jean-Philippe",
            ]
        )

    def test_get_scores_5(self, dartsmodel):
        result = dartsmodel.get_scores(
            [
                "Anas",
                "George",
                "Jean-Philippe",
                "Jean-Philippe",
                "Pierre Edouard",
                "George",
            ]
        )

    def test_get_scores_6(self, dartsmodel):
        result = dartsmodel.get_scores([])
