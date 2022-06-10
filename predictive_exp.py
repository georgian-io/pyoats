from typing import List
import os
import logging
import argparse

from one.utils import *
from one.models import *
from one.data.ucrdata import UcrDataReader

ROOT_DIR = "./data/ucr/"
SAVE_DIR = "./results/"

SIMPLE_MODELS = ["lgbm", "randomforest", "regression"]
DL_MODELS = ["nbeats", "nhits", "tcn", "tft", "transformer"]
RNN_MODELS = ["rnn", "lstm", "gru"]
BASELINE_MODELS = ["iforest"]

FILES = get_files_from_path(ROOT_DIR)


def run_model(m, data, fdir, n_jobs, n_wl, n_hyp):
    print(f"{m.model_name} #####################")
    os.makedirs(fdir, exist_ok=True)

    train_data, train_label = data.train
    test_data, test_label = data.test

    print("Tuning window and step sizes")
    m.hyperopt_ws(train_data, n_wl, n_jobs)

    model_test_data, model_test_label = data.get_test_with_window(m.window)

    print("Tuning model hyperparameters")
    m.hyperopt_model(train_data, n_hyp, n_jobs)
    m.fit(train_data)

    print("Generating predictions and scores")
    score, _, preds = m.get_scores(model_test_data)

    print("Saving outputs")
    save_model_output(score, m, f"{fdir}{m.model_name}_scores.txt")
    save_model_output(preds, m, f"{fdir}{m.model_name}_preds.txt")


def get_data(file: int) -> list:
    reader = UcrDataReader()
    if file == 0:
        return [reader(ROOT_DIR + f) for f in FILES]

    return [reader(ROOT_DIR + FILES[file - 1])]


def if_gpu(if_gpu: str) -> bool:
    return if_gpu.lower() in ["true", "yes", "t", "1"]


def get_models(m: List[str], choices: dict, if_gpu: bool, split: float) -> list:
    models = []

    for model in m:
        model_cls = model_choices.get(model)
        if model in SIMPLE_MODELS:
            models.append(model_cls())
        if model in DL_MODELS:
            models.append(model_cls(use_gpu=if_gpu, val_split=split))
        if model in RNN_MODELS:
            models.append(model_cls(use_gpu=if_gpu, rnn_model=model.upper(), val_split=split))
        if model in BASELINE_MODELS:
            models.append(model_cls())

    return models


def main(data, models, n_jobs, n_wl, n_hyp):
    logger = logging.getLogger()

    for d in data:
        data_name = d.file_name.split(".")[0]
        print(f"{data_name} !!!!!!!!!!!!!!!!!!!!!!")
        fdir = f"{SAVE_DIR}{data_name}/"

        for m in models:
            # LGBM uses all cores by default
            jobs = 1 if isinstance(m, LightGBMModel) else n_jobs
            run_model(m, d, fdir, jobs, n_wl, n_hyp)


if __name__ == "__main__":
    """
    Usage
    ===========
    > python predictive_exp.py -m [--model] M
                               -d [--data] D
                               -s [--split] S
                               -n [--njobs] N
                               -g [--gpu] G
                               -w [--wl] W
                               -y [--hyp] Y

    M -> Literal[ "lightgbm",
                  "randomforest",
                  "regression",
                  "nbeats",
                  "nhits",
                  "tcn",
                  "tft",
                  "transformer",
                  "rnn",
                  "lstm",
                  "gru"]

    D -> Union[int[1:250], Literal["all"]]

    S -> float

    N -> int

    G -> bool

    W -> int

    Y -> int
    """

    parser = argparse.ArgumentParser(description="Run experiment on UCR Dataset")

    model_choices = {
        "lgbm": LightGBMModel,
        "randomforest": RandomForestModel,
        "regression": RegressionModel,
        "nbeats": NBEATSModel,
        "nhits": NHiTSModel,
        "tcn": TCNModel,
        "tft": TFTModel,
        "transformer": TransformerModel,
        "rnn": RNNModel,
        "lstm": RNNModel,
        "gru": RNNModel,
        "iforest": IsolationForestModel,
    }

    parser.add_argument(
        "-m",
        "--model",
        metavar="M",
        choices=model_choices.keys(),
        nargs="+",
        help="One of: lgbm, randomforest, regression, nbeats, nhits, tcn, tft, transformer, rnn, lstm, gru, iforest",
    )

    parser.add_argument(
        "-d",
        "--data",
        metavar="D",
        type=int,
        default=0,
        help="Optional: index # of timeseries from the dataset, default all",
    )

    parser.add_argument(
        "-s",
        "--split",
        metavar="S",
        default=0.1,
        choices=model_choices.keys(),
        type=float,
        help="Optional: Validation split, default 0.1",
    )

    parser.add_argument(
        "-n",
        "--njobs",
        metavar="N",
        default=1,
        type=int,
        help="Optional: number of workers for hyperparameter tuning, default 1",
    )

    parser.add_argument(
        "-g",
        "--gpu",
        metavar="G",
        default="false",
        type=str,
        help="Optional: GPU training (true or false), default false",
    )

    parser.add_argument(
        "-w",
        "--wl",
        metavar="W",
        default=30,
        type=int,
        help="Optional: number of rounds for tuning window & forecast lenght, default 30",
    )

    parser.add_argument(
        "-y",
        "--hyp",
        metavar="Y",
        default=30,
        type=int,
        help="Optional: number of rounds for tuning model hyperparameters, default 30",
    )

    args = parser.parse_args()

    data = get_data(args.data)
    n_jobs = args.njobs
    split = args.split
    if_gpu = if_gpu(args.gpu)
    models = get_models(args.model, model_choices, if_gpu, split)
    n_wl = args.wl
    n_hyp = args.hyp

    main(data, models, n_jobs, n_wl, n_hyp)
