import logging
import os
import sys

import pandas as pd
import yaml
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split

seed = yaml.safe_load(open("params.yaml"))["general"]["seed"]
params = yaml.safe_load(open("params.yaml"))["featurize"]


def featurize(
    clean_data, test_size=params["test_size"], random_state=seed,
):

    test = clean_data.sample(frac=test_size, random_state=random_state)
    train = test.drop(test.index)

    # featurizing steps go here
    featurized_train = train
    featurized_test = test

    return featurized_train, featurized_test


def main():
    """ Runs data processing scripts add features to clean data (saved in data/clean),
        splits the featurized data into training and test datasets and saves them as new
        dataset (in data/featurized)
    """

    logger = logging.getLogger(__name__)
    logger.info("--FEATURIZE--")

    logger.info("Reading arguments")
    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write(
            """
            \tpython featurize.py clean-data featurized-train-data \
            featurized-test-data\n
            """
        )
        sys.exit(1)
    clean_data_path = os.path.join("data", "clean", sys.argv[1])
    featurized_train_path = os.path.join("data", "featurized", sys.argv[2])
    featurized_test_path = os.path.join("data", "featurized", sys.argv[3])

    logger.info(f"Loading clean data from {clean_data_path}")
    clean_data = pd.read_csv(clean_data_path)

    logger.info("Featurizing data")
    featurized_train, featurized_test = featurize(clean_data)

    logger.info(f"Saving featurized training data to {featurized_train_path}")
    featurized_train.to_csv(featurized_train_path, index=False)

    logger.info(f"Saving featurized test data to {featurized_test_path}")
    featurized_test.to_csv(featurized_test_path, index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
