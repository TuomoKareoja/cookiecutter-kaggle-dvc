.PHONY: run clean lint init create_env update_env help
.DEFAULT_GOAL := help

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


##@ Data processing

run: ## Reprocess the needed steps with DVC dag
	dvc repro --pull
	
plot: ## Show DVC plots
	dvc plots show


##@ Testing

lint: ## Lint using flake8
	flake8 src


##@ Setup

init: ## Initialize the project
	git init
	dvc init -q

create_env: ## Create empty Conda enviroment
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	conda create --no-default-packages --yes --name $(PROJECT_NAME) python=3
	@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@echo ">>> Enviroment could not be created. Install conda first"
endif

update_env: ## Update activated Conda environment with enviroment.yml
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, updating activated conda environment."
	conda env update --file environment.yml
	# adding src-files to enviroment for easy importing
	python -m pip install -e .
	@echo ">>> Conda enviroment updated"
else
	@echo ">>> Environment could not be updated. Install conda first"
endif


##@ Helpers

help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

clean: ## Delete all compiled Python files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
