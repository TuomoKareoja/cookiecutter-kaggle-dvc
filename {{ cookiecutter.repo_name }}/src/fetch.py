import logging
import sys
import yaml
import os
import pandas as pd
from dotenv import find_dotenv, load_dotenv

seed = yaml.safe_load(open("params.yaml"))["general"]["seed"]


def fetch_data():

    raw_data = pd.Dataframe()
    return raw_data


def main():
    """ Fetches raw data from external sources and saves it in data/raw.
    """

    logger = logging.getLogger(__name__)
    logger.info("--FETCH--")

    logger.info("Reading arguments")
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython fetch.py raw-output-data\n")
        sys.exit(1)
    raw_data_path = os.path.join("data", "raw", sys.argv[1])

    logger.info("Fetching raw data")
    raw_data = fetch_data()

    logger.info(f"Saving raw data to {raw_data_path}")
    raw_data.to_csv(raw_data_path, index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
