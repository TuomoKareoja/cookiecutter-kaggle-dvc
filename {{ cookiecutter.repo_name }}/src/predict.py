import logging
import os
import sys

import pandas as pd
import yaml
{% if cookiecutter.problem_type == 'Classification' -%}
from catboost import CatBoostClassifier
{% elif cookiecutter.problem_type == 'Regression' -%}
from catboost import CatBoostRegressor
{% endif -%}
from dotenv import find_dotenv, load_dotenv

seed = yaml.safe_load(open("params.yaml"))["general"]["seed"]


def predict(model, featurized_data):

    predictions = model.predict_proba(featurized_data.drop(columns=["target"]))
    {% if cookiecutter.problem_type == 'Classification' -%}
    predictions_proba = model.predict_proba(featurized_data.drop(columns=["target"]))
    {% endif -%}
    predictions_data = featurized_data
    {% if cookiecutter.problem_type == 'Classification' -%}
    predictions_data['pred_proba'] = predictions_proba
    {% endif -%}
    predictions_data['pred'] = predictions

    return predictions_data


def main():
    """ Use pre-trained model(s) to create predictions from the featurized training and
        test datasets (saved in data/featurized) and creates new datasets with the
        predictions added to data/predictions.
    """

    logger = logging.getLogger(__name__)
    logger.info("--PREDICT--")

    logger.info("Reading arguments")
    if len(sys.argv) != 6:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write(
            """
            \tpython predict.py model featurized-train-data \
            featurized-test-data prediction-train-data predictions-test-data\n
            """
        )
        sys.exit(1)
    model_path = os.path.join("models", sys.argv[1])
    featurized_train_path = os.path.join("data", "featurized", sys.argv[2])
    featurized_test_path = os.path.join("data", "featurized", sys.argv[3])
    predictions_train_path = os.path.join("data", "predictions", sys.argv[4])
    predictions_test_path = os.path.join("data", "predictions", sys.argv[5])

    logger.info(f"Loading model from {model_path}")
    {% if cookiecutter.problem_type == 'Classification' -%}
    model = CatBoostClassifier.load_model(model_path)
    {% elif cookiecutter.problem_type == 'Regression' -%}
    model = CatBoostRegressor.load_model(model_path)

    {% endif -%}
    logger.info(f"Loading featurized training data from {featurized_train_path}")
    featurized_train = pd.read_csv(featurized_train_path)

    logger.info(f"Loading featurized test data from {featurized_test_path}")
    featurized_test = pd.read_csv(featurized_test_path)

    logger.info("Creating training predictions")
    predictions_train = predict(model, featurized_train)
    logger.info("Creating test predictions")
    predictions_test = predict(model, featurized_test)

    logger.info(f"Saving training predictions to {predictions_train_path}")
    predictions_train.to_csv(predictions_train_path, index=False)
    logger.info(f"Saving test predictions to {predictions_test_path}")
    predictions_test.to_csv(predictions_test_path, index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
