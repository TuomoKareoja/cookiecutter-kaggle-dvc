.PHONY: run clean lint init create_environment requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = {{ cookiecutter.repo_name }}

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## reprocessed the needed steps in the defined DVC dag
run:
	dvc repro --pull
	
## Show DVC plots
plot:
	dvc plots show

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Initialize the project
init:
	git init
	dvc init -q

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	conda create --no-default-packages --yes --name $(PROJECT_NAME) python=3
	@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@echo ">>> Enviroment could not be created. Install conda first"
endif

## Install Python Dependencies
requirements:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, updating activated conda environment."
	conda env update --file environment.yml
	# adding src-files to enviroment for easy importing
	python -m pip install -e .
	@echo ">>> Conda enviroment updated"
else
	@echo ">>> Environment could not be updated. Install conda first"
endif

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# TODO add better command