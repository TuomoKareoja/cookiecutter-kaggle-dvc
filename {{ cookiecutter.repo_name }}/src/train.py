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
params = yaml.safe_load(open("params.yaml"))["train"]


def train(featurized_data, params=params, seed=seed):

    {% if cookiecutter.problem_type == 'Classification' -%}
    model = CatBoostClassifier(
        iterations=params["iterations"],
        depth=params["depth"],
        l2_leaf_reg=params["l2_leaf_reg"],
        random_strength=params["random_strength"],
        bagging_temperature=params["bagging_temperature"],
        loss_function=params["loss_function"],
        random_seed=seed,
    )
    {% elif cookiecutter.problem_type == 'Regression' -%}
    model = CatBoostRegressor(
        iterations=params["iterations"],
        depth=params["depth"],
        l2_leaf_reg=params["l2_leaf_reg"],
        random_strength=params["random_strength"],
        bagging_temperature=params["bagging_temperature"],
        loss_function=params["loss_function"],
        random_seed=seed,
    )
    {% endif -%}
    model.fit(featurized_data.drop(columns=["target"]), featurized_data["target"])

    return model


def main():
    """ Train model(s) with the featurized training dataset (saved in data/featurized)
    """

    logger = logging.getLogger(__name__)
    logger.info("--TRAIN--")

    logger.info("Reading arguments")
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write(
            """
            \tpython train.py featurized-train-data model\n
            """
        )
        sys.exit(1)
    featurized_train_path = os.path.join("data", "featurized", sys.argv[1])
    model_path = os.path.join("models", sys.argv[2])

    logger.info(f"Loading featurized training data from {featurized_train_path}")
    featurized_train = pd.read_csv(featurized_train_path)

    logger.info("Training the model")
    model = train(featurized_train)

    logger.info(f"Saving the the trained model to {model_path}")
    model.save_model(model_path)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
