import os
import json
import uuid
from datetime import datetime
from pathlib import Path

import typer
import joblib
import shap
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import threading

__version__ = "0.1.0"

# Configuration
DECKARD_HOME = Path.home() / ".deckard"
MODELS_DIR = DECKARD_HOME / "models"
ARTIFACTS_DIR = DECKARD_HOME / "artifacts"
DATA_DIR = DECKARD_HOME / "data"

# Ensure directories exist
DECKARD_HOME.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

app = typer.Typer(
    help="Deckard CLI - Develop, manage, deploy, and monitor machine learning models."
)

@app.command()
def version():
    """Show the version of Deckard CLI."""
    typer.echo(f"Deckard {__version__}")

@app.command()
def artifacts(run_id: str = typer.Option(..., help="Run ID to list artifacts for")):
    """List artifacts for a given run ID."""
    list_artifacts(run_id)

@app.command()
def explain(model_id: str = typer.Option(..., help="Model ID to explain")):
    """Generate SHAP explanations for a given model ID."""
    explain_model(model_id)

@app.command()
def train(
    task: str = typer.Option("classification", help="Task type: classification or regression"),
    n_samples: int = typer.Option(1000, help="Number of samples to generate"),
    n_features: int = typer.Option(10, help="Number of features"),
    local: bool = typer.Option(True, help="Train the model locally")
):
    """Train a model locally or remotely."""
    train_model(task, n_samples, n_features, local)

@app.command()
def serve(
    model_id: str = typer.Option(..., help="Model ID to serve"),
    port: int = typer.Option(5000, help="Port to serve the model on")
):
    """Serve a trained model on a specified port."""
    serve_model(model_id, port)

@app.command()
def list_models():
    """List all trained models."""
    models = list(MODELS_DIR.glob("*.pkl"))
    if not models:
        typer.echo("No models found.")
        return
    
    typer.echo("Available models:")
    for model_path in models:
        model_id = model_path.stem
        metadata_path = ARTIFACTS_DIR / f"{model_id}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                typer.echo(f"  {model_id} - {metadata.get('task', 'unknown')} ({metadata.get('timestamp', 'unknown time')})")
        else:
            typer.echo(f"  {model_id}")

def list_artifacts(run_id):
    """List all artifacts associated with a specific run ID.
    
    Args:
        run_id (str): The ID of the run to list artifacts for
    """
    typer.echo(f"Listing artifacts for run ID: {run_id}")
    
    # Check if artifacts exist for this run ID
    artifacts = list(ARTIFACTS_DIR.glob(f"{run_id}*"))
    
    if not artifacts:
        typer.echo(f"No artifacts found for run ID: {run_id}")
        return
    
    typer.echo("Found artifacts:")
    for artifact in artifacts:
        typer.echo(f"  {artifact.name}")

def load_data_for_model(model_id):
    """Load the data associated with a specific model.
    
    Args:
        model_id (str): The ID of the model to load data for
        
    Returns:
        pandas.DataFrame: The loaded data for model explanation
    """
    data_path = DATA_DIR / f"{model_id}_data.csv"
    
    if data_path.exists():
        typer.echo(f"Loading data for model: {model_id}")
        return pd.read_csv(data_path)
    else:
        # Generate sample data for demonstration
        typer.echo(f"No data file found for model {model_id}, generating sample data")
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        return data

def explain_model(model_id):
    """Generate and visualize SHAP explanations for the specified model.
    
    Args:
        model_id (str): The ID of the model to explain
    """
    model_path = MODELS_DIR / f"{model_id}.pkl"
    
    if not model_path.exists():
        typer.echo(f"Model {model_id} not found!")
        return
    
    typer.echo(f"Generating SHAP explanations for model ID: {model_id}")
    
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Load the data for which we want to explain the predictions
        data = load_data_for_model(model_id)
        
        # Create a SHAP explainer
        explainer = shap.Explainer(model)
        
        # Calculate SHAP values for a subset of data
        sample_data = data.head(10)  # Use first 10 rows for explanation
        shap_values = explainer(sample_data)
        
        # Save SHAP plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, sample_data, show=False)
        
        # Save the plot
        plot_path = ARTIFACTS_DIR / f"{model_id}_shap_plot.png"
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        typer.echo(f"SHAP explanation plot saved to: {plot_path}")
        
    except Exception as e:
        typer.echo(f"Error generating SHAP explanations: {str(e)}")

def train_model(task, n_samples, n_features, local):
    """Train a machine learning model either locally or remotely.
    
    Args:
        task (str): Type of task - classification or regression
        n_samples (int): Number of samples to generate
        n_features (int): Number of features
        local (bool): If True, trains the model locally. If False, trains remotely.
    """
    if not local:
        typer.echo("Remote training not implemented in v0. Training locally instead.")
    
    typer.echo(f"Training {task} model locally with {n_samples} samples and {n_features} features")
    
    # Generate a unique model ID
    model_id = f"model_{uuid.uuid4().hex[:8]}"
    
    try:
        # Generate synthetic data
        if task == "classification":
            # Ensure n_informative is at least 2 for binary classification and doesn't exceed n_features
            n_informative = min(max(2, n_features), n_features)
            X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                                     n_informative=n_informative, n_redundant=0, 
                                     n_clusters_per_class=1, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif task == "regression":
            X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                                 noise=0.1, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            typer.echo(f"Unsupported task type: {task}")
            return
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if task == "classification":
            score = accuracy_score(y_test, y_pred)
            metric_name = "accuracy"
        else:
            score = mean_squared_error(y_test, y_pred)
            metric_name = "mse"
        
        # Save the model
        model_path = MODELS_DIR / f"{model_id}.pkl"
        joblib.dump(model, model_path)
        
        # Save the data
        data_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        data_path = DATA_DIR / f"{model_id}_data.csv"
        data_df.to_csv(data_path, index=False)
        
        # Save metadata
        metadata = {
            "model_id": model_id,
            "task": task,
            "n_samples": n_samples,
            "n_features": n_features,
            "timestamp": datetime.now().isoformat(),
            metric_name: score,
            "model_type": type(model).__name__
        }
        
        metadata_path = ARTIFACTS_DIR / f"{model_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        typer.echo(f"Model trained successfully!")
        typer.echo(f"Model ID: {model_id}")
        typer.echo(f"Model saved to: {model_path}")
        typer.echo(f"{metric_name.upper()}: {score:.4f}")
        
    except Exception as e:
        typer.echo(f"Error training model: {str(e)}")

def serve_model(model_id, port):
    """Serve a trained model via a REST API.
    
    Args:
        model_id (str): The ID of the model to serve
        port (int): The port number to serve the model on
    """
    model_path = MODELS_DIR / f"{model_id}.pkl"
    
    if not model_path.exists():
        typer.echo(f"Model {model_id} not found!")
        return
    
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Create Flask app
        flask_app = Flask(__name__)
        
        @flask_app.route('/predict', methods=['POST'])
        def predict():
            try:
                data = request.json
                features = np.array(data['features']).reshape(1, -1)
                prediction = model.predict(features)
                return jsonify({'prediction': prediction.tolist()})
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        @flask_app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'healthy', 'model_id': model_id})
        
        typer.echo(f"Serving model {model_id} on port {port}")
        typer.echo(f"Health check: http://localhost:{port}/health")
        typer.echo(f"Prediction endpoint: http://localhost:{port}/predict")
        typer.echo("Press Ctrl+C to stop the server")
        
        # Run the Flask app
        flask_app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        typer.echo(f"Error serving model: {str(e)}")

if __name__ == "__main__":
    app()
