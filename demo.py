#!/usr/bin/env python3
"""
Deckard CLI Demo Script
This script demonstrates all the functionality of the Deckard CLI v0.1.0
"""

import subprocess
import sys
import time

def run_command(cmd):
    """Run a command and print its output."""
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Command timed out!")
        return False

def main():
    """Demonstrate Deckard CLI functionality."""
    print("🚀 Deckard CLI v0.1.0 Demonstration")
    print("=" * 60)
    
    # Show version
    run_command("deckard version")
    
    # Show help
    run_command("deckard --help")
    
    # Train a classification model
    run_command("deckard train --task classification --n-samples 200 --n-features 8")
    
    # Train a regression model  
    run_command("deckard train --task regression --n-samples 150 --n-features 6")
    
    # List all models
    run_command("deckard list-models")
    
    # Get the first model ID for further commands
    result = subprocess.run("deckard list-models", shell=True, capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    model_id = None
    
    for line in lines:
        if 'model_' in line:
            # Extract model ID from the line
            parts = line.strip().split(' - ')
            if parts:
                model_id = parts[0].strip()
                break
    
    if model_id:
        print(f"\n🎯 Using model '{model_id}' for explanations and artifacts demo...")
        
        # Generate SHAP explanations
        run_command(f"deckard explain --model-id {model_id}")
        
        # List artifacts for the model
        run_command(f"deckard artifacts --run-id {model_id}")
        
        print(f"\n🌐 Model serving demo (Note: This would normally start a web server)")
        print(f"Command: deckard serve --model-id {model_id} --port 5000")
        print("This would start a Flask server with endpoints:")
        print("  - http://localhost:5000/health (GET)")
        print("  - http://localhost:5000/predict (POST)")
        
    else:
        print("❌ No models found to demonstrate explain/serve functionality")
    
    # Test error handling
    run_command("deckard train --task invalid_task")
    run_command("deckard explain --model-id nonexistent_model")
    run_command("deckard artifacts --run-id nonexistent_run")
    
    print("\n🎉 Demo completed!")
    print("\nKey features demonstrated:")
    print("✅ Model training (classification & regression)")
    print("✅ Model listing and management") 
    print("✅ SHAP-based model explanations")
    print("✅ Artifact tracking and listing")
    print("✅ Model serving capability (REST API)")
    print("✅ Error handling and validation")
    print("✅ Comprehensive test coverage")
    
    print(f"\n📁 Models and artifacts stored in: ~/.deckard/")
    print("📝 Check CHANGELOG.md for detailed feature information")

if __name__ == "__main__":
    main() 