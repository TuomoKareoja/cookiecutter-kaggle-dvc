import json
import logging
import os
import sys

import pandas as pd
{% if cookiecutter.problem_type == 'Regression' -%}
import numpy as np
{% endif -%}
import sklearn.metrics as metrics
import yaml
from dotenv import find_dotenv, load_dotenv

seed = yaml.safe_load(open("params.yaml"))["general"]["seed"]


{% if cookiecutter.problem_type == 'Classification' -%}
def score(
    scores_path, target_train, target_test, pred_proba_train, pred_proba_test
):

    avg_prec_train = metrics.average_precision_score(target_train, pred_proba_train)
    avg_prec_test = metrics.average_precision_score(target_train, pred_proba_test)
    roc_auc_train = metrics.roc_auc_score(target_train, pred_proba_train)
    roc_auc_test = metrics.roc_auc_score(target_test, pred_proba_test)

    scores = {
        "train": {"avg_precision": avg_prec_train, "roc_auc": roc_auc_train},
        "test": {"avg_precision": avg_prec_test, "roc_auc": roc_auc_test},
        "difference": {
            "avg_precision": avg_prec_test - avg_prec_train,
            "roc_auc": roc_auc_test - roc_auc_train,
        },
    }

    with open(scores_path, "w") as scores_file:
        json.dump(scores, scores_file, indent=4)


def plot(
    prc_path, roc_path, confusion_path, target, pred, pred_proba,
):

    precision, recall, prc_thresholds = metrics.precision_recall_curve(
        target, pred_proba
    )
    fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)

    with open(prc_path, "w") as prc_file:
        json.dump(
            {
                "prc": [
                    {"precision": p, "recall": r, "threshold": t}
                    for p, r, t in zip(precision, recall, prc_thresholds)
                ]
            },
            prc_file,
            indent=4,
        )

    with open(roc_path, "w") as roc_file:
        json.dump(
            {
                "roc": [
                    {"fpr": fp, "tpr": tp, "threshold": t}
                    for fp, tp, t in zip(fpr, tpr, roc_thresholds)
                ]
            },
            roc_file,
            indent=4,
        )

    with open(confusion_path, "w") as confusion_file:
        json.dump(
            {
                "confusion": [
                    {"target": target, "prediction": pred}
                    for target, pred in zip(target, pred)
                ]
            },
            confusion_file,
            indent=4,
        )
{% elif cookiecutter.problem_type == 'Regression' -%}
def score(
    scores_path, target_train, target_test, pred_train, pred_test
):

    mean_squared_error_train = metrics.mean_squared_error(target_train, pred_train)
    mean_squared_error_test = metrics.mean_squared_error(target_test, pred_test)
    mean_absolute_error_train = metrics.mean_absolute_error(target_train, pred_train)
    mean_absolute_error_test = metrics.mean_absolute_error(target_test, pred_test)
    r2_train = metrics.r2_score(target_train, pred_train)
    r2_test = metrics.r2_score(target_test, pred_test)

    scores = {
        "train": {
            "mean_squared_error": mean_squared_error_train,
            "mean_absolute_error": mean_absolute_error_train,
            "R2": r2_train
        },
        "test": {
            "mean_squared_error": mean_squared_error_test,
            "mean_absolute_error": mean_absolute_error_test,
            "R2": r2_test
        },
        "difference": {
            "mean_squared_error": mean_squared_error_train - mean_squared_error_test,
            "mean_absolute_error": mean_absolute_error_train - mean_absolute_error_test,
            "R2": r2_test - r2_train
        }
    }

    with open(scores_path, "w") as scores_file:
        json.dump(scores, scores_file, indent=4)


def plot(residuals_path, target, pred, seed=seed):

    np.random.seed(seed)
    # for bigger datasets we only show 5000 dots
    sample_size = min(len(target), 5000)
    sample_idx = np.random.randint(sample_size, size=len(target)).astype('bool')
    target_sample = target[sample_idx]
    pred_sample = pred[sample_idx]

    with open(residuals_path, "w") as residuals_file:
        json.dump(
            {
                "residuals": [
                    {"target": target, "prediction": pred}
                    for target, pred in zip(target_sample, pred_sample)
                ]
            },
            residuals_file,
            indent=4,
        )
{% endif %}

def main():
    """ Evaluates model predictions for both training and test
        prediction datasets (saved in data/predictions)
    """

    logger = logging.getLogger(__name__)
    logger.info("--EVALUATE--")

    logger.info("Reading arguments")
    {% if cookiecutter.problem_type == 'Classification' -%}
    if len(sys.argv) != 10:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write(
            """
            \tpython evaluate.py predictions-train-data predictions-test-data \
            scores-file pcr-train-file pcr-test-file roc-train-file roc-test-file \
            confusion-train-file confusion-test-file\n
            """
        )
        sys.exit(1)
    predictions_path_train = os.path.join("data", "predictions", sys.argv[1])
    predictions_path_test = os.path.join("data", "predictions", sys.argv[2])
    scores_path = sys.argv[3]
    prc_path_train = sys.argv[4]
    prc_path_test = sys.argv[5]
    roc_path_train = sys.argv[6]
    roc_path_test = sys.argv[7]
    confusion_path_train = sys.argv[8]
    confusion_path_test = sys.argv[9]

    logger.info(f"Loading training predictions data from {predictions_path_train}")
    {% elif cookiecutter.problem_type == 'Regression' -%}
    if len(sys.argv) != 6:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write(
            """
            \tpython evaluate.py predictions-train-data predictions-test-data \
            scores-file residuals-train-file residuals-test-file\n
            """
        )
        sys.exit(1)
    predictions_path_train = os.path.join("data", "predictions", sys.argv[1])
    predictions_path_test = os.path.join("data", "predictions", sys.argv[2])
    scores_path = sys.argv[3]
    residuals_path_train = sys.argv[4]
    residuals_path_test = sys.argv[5]

    logger.info(f"Loading training predictions data from {predictions_path_train}")
    {% endif -%}
    predictions_train = pd.read_csv(predictions_path_train)

    logger.info(f"Loading test predictions data from {predictions_path_test}")
    predictions_test = pd.read_csv(predictions_path_test)

    target_train = predictions_train["target"]
    target_test = predictions_test["target"]
    {% if cookiecutter.problem_type == 'Classification' -%}
    pred_proba_train = predictions_train["pred_proba"]
    pred_proba_test = predictions_test["pred_proba"]
    {% endif -%}
    pred_train = predictions_train["pred"]
    pred_test = predictions_test["pred"]

    {% if cookiecutter.problem_type == 'Classification' -%}
    logger.info("Calculating and saving scores")
    score(
        scores_path,
        target_train,
        target_test,
        pred_proba_train,
        pred_proba_test
    )

    logger.info("Creating evaluation plots")
    plot(
        prc_path_train,
        roc_path_train,
        confusion_path_train,
        target_train,
        pred_train,
        pred_proba_train
    )
    plot(
        prc_path_test,
        roc_path_test,
        confusion_path_test,
        target_test,
        pred_test,
        pred_proba_test,
    {% elif cookiecutter.problem_type == 'Regression' -%}
    logger.info("Calculating and saving scores")
    score(
        scores_path, target_train, target_test, pred_train, pred_test
    )

    logger.info("Creating evaluation plots")
    plot(
        residuals_path_train, target_train, pred_train
    )
    plot(
        residuals_path_test, target_test, pred_test
    {% endif -%}
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
