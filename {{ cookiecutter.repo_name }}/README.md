{{cookiecutter.project_name}}
==============================

{{cookiecutter.description}}

Project Organization
------------

    ├── data
    │   ├── clean          <- Cleaned datasets
    │   ├── featurized     <- Featurized datasets
    │   ├── predictions    <- Datasets with predictions included for exploration
    │   ├── raw            <- The original, immutable data dump
    │   └── submissions    <- Kaggle-ready submission files
    │
    ├── notebooks
    │   ├── presentation   <- Cleaned up notebooks for future reference
    │   └── sketch         <- Working notebooks and messy tests
    │
    ├── src                <- Source code for use in this project
    │   ├── misc           <- For functions used in exploration and testing
    │   │   └─ __init__.py
    │   ├── __init__.py
    │   ├── clean          <- DVC script for cleaning the data
    │   ├── evaluate       <- DVC script for evaluating, saving predictions, and creating a submission
    │   └── featurize      <- DVC script for featurizing the cleaned data
    │
    ├── .dvcignore         <- Marks files for DVC to ignore
    ├── .env               <- Non-tracked file for holding private passwords and keys
    ├── .gitignore
    ├── dvc.yaml           <- Defines DVC dag steps. These steps use the files on the root of src
    ├── environment.yml    <- Conda environment file for reproducing the analysis environment
    ├── LICENSE
    ├── Makefile           <- Makefile with commands for setting up the project and running DVC
    ├── params.yaml        <- Hyperparameters used in the DVC dag
    ├── README.md
    ├── setup.py           <- makes project pip installable (pip install -e .) so that src can be imported
    └── setup.cfg          <- Python configuration file (e.g. Flake8)

--------
