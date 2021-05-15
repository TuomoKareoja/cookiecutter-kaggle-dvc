# Cookiecutter for Kaggle Competitions

_A logical, reasonably standardized project structure built around DVC for doing and sharing data science_

## Requirements to use the cookiecutter template:
-----------
 - Python 3.5 and up
 - [Cookiecutter Python package](http://cookiecutter.readthedocs.org/en/latest/installation.html) >= 1.4.0: This can be installed with pip by or conda depending on how you manage your Python packages:

``` bash
$ pip install cookiecutter
```

or

``` bash
$ conda config --add channels conda-forge
$ conda install cookiecutter
```


## To start a new project, run:
------------

    cookiecutter https://github.com/TuomoKareoja/cookiecutter-kaggle-dvc



The resulting directory structure for your new project will look like this:

```
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
```

## Installing development requirements
------------

### Prerequisites

 - Conda

### Steps

1. Create enviroment
```
    make create_environment
```

2. activate created environment with conda or virtualenv
```
    conda activate cookiecutter-kaggle-dvc
```

3. Install required packages
```
    make requirements
```
