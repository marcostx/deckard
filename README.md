
![b385c387-bladerunner](https://github.com/user-attachments/assets/210ba251-3a1d-4a43-a419-b1e5b1887054)

# deckard
Deckard is a command-line interface (CLI) library designed to help you develop, manage, deploy, and monitor machine-learning models. With Deckard, you can easily train models locally, generate SHAP explanations, list artifacts, and serve models via a web server.

## Features

- Train machine learning models locally
- Generate SHAP explanations for model interpretability
- List artifacts associated with specific run IDs
- Serve trained models via a web server

## Installation

Deckard is designed to be installed in a conda environment. Follow these steps to set up the environment and install the CLI:

1. **Set up the conda environment and install the CLI:**
   ```bash
   make setup
   ```

   This command will create a conda environment named `deckard_env`, install the necessary dependencies, and install the Deckard CLI.

2. **Activate the conda environment:**
   ```bash
   conda activate deckard_env
   ```

3. **Generate local documentation:**
   Navigate to the `docs` directory and run the following command to generate the HTML documentation:
   ```bash
   make html
   ```

   The generated documentation can be found in the `_build/html` directory.

4. **Run tests to verify everything works:**
   ```bash
   python -m pytest tests/ -v
   ```

## Usage

### Quick Start

After installation, you can immediately start using Deckard:

```bash
# Show version
deckard version

# Train a classification model
deckard train --task classification --n-samples 1000 --n-features 8

# Train a regression model
deckard train --task regression --n-samples 500 --n-features 5

# List all trained models
deckard list-models

# Generate SHAP explanations for a model
deckard explain --model-id <your-model-id>

# List artifacts for a model
deckard artifacts --run-id <your-model-id>

# Serve a model via REST API
deckard serve --model-id <your-model-id> --port 5000
```

### Available Commands

- `deckard version` - Show version information
- `deckard train` - Train classification or regression models
- `deckard list-models` - List all available trained models
- `deckard explain` - Generate SHAP explanations for model interpretability
- `deckard artifacts` - List artifacts associated with specific run IDs
- `deckard serve` - Serve trained models via REST API

### Demo

Run the included demo script to see all features in action:

```bash
python demo.py
```

### REST API Endpoints

When serving a model, the following endpoints are available:

- `GET /health` - Health check endpoint
- `POST /predict` - Make predictions with JSON payload: `{"features": [1, 2, 3, ...]}`

Use the `deckard` command in the terminal to access the CLI features. Run `deckard --help` to see the available commands and options.

## License

Deckard is licensed under the MIT License.
