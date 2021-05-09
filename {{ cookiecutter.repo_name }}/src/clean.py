import logging
import os
import sys

import pandas as pd
import yaml
from dotenv import find_dotenv, load_dotenv

seed = yaml.safe_load(open("params.yaml"))["general"]["seed"]


def clean(raw_data):

    # cleaning steps go here
    clean_data = raw_data

    return clean_data


def main():
    """ Runs data processing scripts to turn raw data from (data/raw) into
        cleaned data ready to be analyzed (saved in data/clean).
    """

    logger = logging.getLogger(__name__)
    logger.info("--CLEAN--")

    logger.info("Reading arguments")
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write(
            """
            \tpython clean.py raw-data clean-data\n
            """
        )
        sys.exit(1)
    raw_data_path = os.path.join("data", "raw", sys.argv[1])
    clean_data_path = os.path.join("data", "clean", sys.argv[2])

    logger.info(f"Loading raw data from {raw_data_path}")
    raw_data = pd.read_csv(raw_data_path)

    logger.info("Cleaning data")
    clean_data = clean(raw_data)

    logger.info(f"Saving cleaned data to {clean_data_path}")
    clean_data.to_csv(clean_data_path, index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
