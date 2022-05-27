import os
import logging

from one.utils import *
from one.models import *
from one.data.ucrdata import UcrDataReader

ROOT_DIR = "./data/ucr/"
SAVE_DIR = "./results/"

SIMPLE_MODELS = [LightGBMModel, RandomForestModel, RegressionModel]
DL_MODELS = [NBEATSModel, NHiTSModel, TCNModel, TFTModel, TransformerModel]
RNN_MODELS = ["RNN", "LSTM", "GRU"]

FILES = get_files_from_path(ROOT_DIR)


def run_model(m, data, fdir):
    print(f"{m.model_name} #####################")
    os.makedirs(fdir, exist_ok=True)

    train_data, train_label = data.train
    test_data, test_label = data.test

    print("Tuning window and step sizes")
    m.hyperopt_ws(train_data, test_data, 35)

    model_test_data, model_test_label = data.get_test_with_window(m.window)

    print("Tuning model hyperparameters")
    m.hyperopt_model(train_data, model_test_data, 45)
    m.fit(train_data)

    print("Generating predictions and scores")
    score, _, preds = m.get_scores(model_test_data)

    print("Saving outputs")
    save_model_output(score, m, f"{fdir}{m.model_name}_scores.txt")
    save_model_output(preds, m, f"{fdir}{m.model_name}_preds.txt")


def main():
    logger = logging.getLogger()
    reader = UcrDataReader()

    for file in FILES:
        print(f"{file} !!!!!!!!!!!!!!!!!!!!!!")
        data_name = file.split(".")[0]
        data = reader(ROOT_DIR + file)
        fdir = f"{SAVE_DIR}{data_name}/"
        
        for model in SIMPLE_MODELS:
            m = model()
            run_model(m, data, fdir)

        for model in DL_MODELS:
            m = model(use_gpu=True, val_split=0.1)
            run_model(m, data, fdir)

        for model in RNN_MODELS:
            m = RNNModel(use_gpu=True, rnn_model=model, val_split=0.1)
            run_model(m, data, fdir)


if __name__ == "__main__":
    main()
