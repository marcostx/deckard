
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

## Usage

Use the `deckard` command in the terminal to access the CLI features. Run `deckard --help` to see the available commands and options.

## License

Deckard is licensed under the MIT License.
