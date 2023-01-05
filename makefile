SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:

MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

COMMA := ,

PROJECT_NAME := $(shell echo $(notdir $(CURDIR)) | sed -e 's/_/-/g')
PYTHON_IMAGE_VERSION := $(shell cat .python-version)

# PIP_COMPILE := pip-compile --generate-hashes
PIP_COMPILE := pip-compile

venv:
	@python -m venv .venv
	./.venv/bin/pip install -U pip pip-tools
	@echo -e "\nINFO: virtualenv create success at '.venv'."
.PHONY: venv

pip: requirements/base.in $(shell ls requirements/*.in | grep -v "base.in")  ## Generate requirements
	$(foreach F,$^,$(PIP_COMPILE) $(F) &&) echo "Compile success."
.PHONY: pip

pip-clean: requirements/*.txt  ## Remove requirements output file
	rm -rvf $^
.PHONY: pip-clean

install: requirements/base.txt $(foreach M,$(subst $(COMMA), ,$(mode)),requirements/$(M).txt)  ## Install depends packages, `mode` with params
	pip-sync $^
.PHONY: install

lint:  ## Check lint
	isort --check --diff .
	black --check --diff .
.PHONY: lint

lint-fix: ## Fix lint
	isort .
	black .
.PHONY: lint-fix

clean:  ## Clean cache files
	find . -name '__pycache__' -type d | xargs rm -rvf
	find . -name '.mypy_cache' -type d | xargs rm -rvf
	find . -name '.pytest_cache' -type d | xargs rm -rvf
.PHONY: clean


notebook:  ## Run Jupyter Lab
	jupyter-lab --allow-root --ip=0.0.0.0 --no-browser
.PHONY: notebook

runserver:  ## Run streamlit
	streamlit run main.py --server.port 8088
.PHONY: runserver

.DEFAULT_GOAL := help
help: Makefile
	@grep -E '(^[a-zA-Z_-]+:.*?##.*$$)|(^##)' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[32m%-30s\033[0m %s\n", $$1, $$2}' | sed -e 's/\[32m##/[33m/'
.PHONY: help
