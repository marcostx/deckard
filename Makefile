CONDA_ENV_NAME := deckard_env

PYTHON_VERSION := 3.8

.PHONY: setup clean

setup: create_env install_deps install_cli

create_env:
	@echo "Creating conda environment $(CONDA_ENV_NAME)..."
	@conda create --name $(CONDA_ENV_NAME) python=$(PYTHON_VERSION) -y

install_deps:
	@echo "Installing dependencies..."
	@conda run -n $(CONDA_ENV_NAME) pip install -r requirements.txt

install_cli:
	@echo "Installing Deckard CLI..."
	@conda run -n $(CONDA_ENV_NAME) pip install -e .

clean:
	@echo "Removing conda environment $(CONDA_ENV_NAME)..."
	@conda env remove -n $(CONDA_ENV_NAME) -y

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make setup  - Create conda environment, install dependencies, and install CLI"
	@echo "  make clean  - Remove the conda environment"
	@echo "  make help   - Show this help message"