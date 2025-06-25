# Changelog

All notable changes to the Deckard CLI project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-XX - Functional v0 Release

### Added

#### Core Features
- **Model Training**: Implemented functional training for both classification and regression tasks
  - Support for synthetic data generation with configurable samples and features
  - Random Forest models as default algorithms
  - Automatic model persistence using joblib
  - Performance metrics calculation (accuracy for classification, MSE for regression)
  - Unique model ID generation for tracking

- **Model Serving**: Added REST API serving functionality
  - Flask-based web server for model deployment
  - `/predict` endpoint for making predictions via POST requests
  - `/health` endpoint for service health checks
  - Configurable port selection
  - JSON-based request/response handling

- **Model Explanation**: Implemented SHAP-based model interpretability
  - SHAP explainer integration for model explanations
  - Summary plot generation and saving
  - Automatic data loading for explanations
  - PNG output for visualization artifacts

- **Artifact Management**: Added comprehensive artifact tracking
  - Model metadata storage in JSON format
  - Artifact listing by run/model ID
  - Organized storage in `~/.deckard/` directory structure
  - Training data persistence for reproducibility

- **Model Management**: Added model listing and organization
  - `list-models` command to show all trained models
  - Metadata display including task type and timestamps
  - Organized storage in models, artifacts, and data directories

#### CLI Commands
- `deckard version` - Show version information
- `deckard train` - Train classification or regression models
  - `--task` option for model type (classification/regression)
  - `--n-samples` option for dataset size
  - `--n-features` option for feature count
  - `--local` flag for local training (remote not implemented in v0)
- `deckard serve` - Serve trained models via REST API
  - `--model-id` option to specify model
  - `--port` option for server port
- `deckard explain` - Generate SHAP explanations
  - `--model-id` option to specify model
- `deckard artifacts` - List artifacts for run ID
  - `--run-id` option to specify run
- `deckard list-models` - List all available trained models

#### Development Infrastructure
- **Testing Suite**: Comprehensive test coverage using pytest
  - Unit tests for all CLI commands
  - Integration tests for end-to-end workflows
  - Mocking for external dependencies (SHAP, Flask, matplotlib)
  - Temporary directory management for isolated tests
  - Test fixtures for consistent test environments

- **Dependencies**: Complete dependency management
  - Updated `requirements.txt` with all necessary packages
  - Version pinning for stability
  - Test dependencies separated in setup.py extras_require

- **Project Structure**: Organized codebase structure
  - Proper package discovery in setup.py
  - Source code organization in `src/deckard/`
  - Test organization in `tests/`
  - Documentation structure maintained

### Changed

#### Configuration
- **Setup Configuration**: Enhanced setup.py with proper package discovery
  - Added missing setuptools imports
  - Fixed package path configuration for src layout
  - Added comprehensive dependency specifications
  - Included test dependencies as extras

- **Storage System**: Implemented organized file storage
  - Models stored in `~/.deckard/models/`
  - Artifacts stored in `~/.deckard/artifacts/`
  - Training data stored in `~/.deckard/data/`
  - Automatic directory creation on first use

#### CLI Interface
- **Enhanced Train Command**: Expanded training options
  - Added task type selection (classification/regression)
  - Configurable dataset parameters
  - Improved output with model ID and metrics
  - Better error handling and user feedback

- **Improved User Experience**: Better command-line interaction
  - Consistent use of typer.echo for output
  - Clear error messages
  - Progress indicators and success confirmations
  - Helpful command descriptions and help text

### Fixed

#### Core Issues
- **Import Problems**: Resolved missing imports in setup.py
- **Package Structure**: Fixed package discovery for proper installation
- **Dependency Management**: Complete dependency specification
- **CLI Functionality**: All placeholder TODO implementations replaced with working code

#### Error Handling
- **Model Loading**: Proper validation of model existence
- **File Operations**: Safe file handling with error messages
- **Command Validation**: Input validation with helpful error messages

### Technical Details

#### Architecture
- **Modular Design**: Separated concerns with distinct functions for each operation
- **Configuration Management**: Centralized path configuration with environment variables
- **Error Handling**: Comprehensive exception handling throughout the codebase

#### Dependencies
- **Core Dependencies**: typer, joblib, pandas, numpy, scikit-learn
- **ML/AI Libraries**: shap for explainability, matplotlib for visualization
- **Web Framework**: Flask for model serving
- **Testing**: pytest with mocking capabilities

#### Storage Format
- **Models**: Pickled using joblib (.pkl files)
- **Metadata**: JSON format for human-readable artifact information
- **Data**: CSV format for training data persistence
- **Plots**: PNG format for SHAP visualization exports

### Notes for Users

#### Getting Started
1. Install the package: `pip install -e .` (or use `make setup`)
2. Train your first model: `deckard train --task classification`
3. List available models: `deckard list-models`
4. Serve a model: `deckard serve --model-id <your-model-id>`
5. Generate explanations: `deckard explain --model-id <your-model-id>`

#### Limitations in v0
- Remote training is not implemented (falls back to local training)
- Only synthetic data generation is available (no external data loading)
- Limited to Random Forest algorithms
- SHAP explanations limited to first 10 samples
- Basic REST API without authentication

#### Future Enhancements (Planned)
- Support for custom datasets
- Additional ML algorithms
- Remote training capabilities
- Enhanced visualization options
- Authentication for serving endpoints
- Batch prediction capabilities

[0.1.0]: https://github.com/marcostx/deckard/releases/tag/v0.1.0 