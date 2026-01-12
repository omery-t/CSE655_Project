"""
Inference script for making predictions with saved models.
Loads pre-trained models without retraining.

Usage:
    python scripts/inference.py                     # Use default saved model
    python scripts/inference.py --model rf          # Use specific model type
    python scripts/inference.py --input data.csv   # Predict on custom data
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODELS_DIR, DATA_FILE, BASE_DIR
from data_loader import load_data, preprocess_data, get_feature_names

sys.path.insert(0, os.path.dirname(BASE_DIR))
from helpers.report_manager import update_prediction_section
from helpers.logger import log


def load_saved_model(model_name='rf', models_dir=None):
    """
    Load a saved model from disk.
    
    Args:
        model_name: Name of the model ('ann', 'svm', 'xgboost', 'rf')
        models_dir: Directory containing saved models
        
    Returns:
        Loaded model object
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    
    model_path = os.path.join(models_dir, f'{model_name}_model.joblib')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Please train the model first using: python scripts/train.py"
        )
    
    log.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    return model


def list_available_models(models_dir=None):
    """
    List all available saved models.
    
    Args:
        models_dir: Directory containing saved models
        
    Returns:
        List of available model names
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    
    if not os.path.exists(models_dir):
        return []
    
    models = []
    for file in os.listdir(models_dir):
        if file.endswith('_model.joblib'):
            model_name = file.replace('_model.joblib', '')
            models.append(model_name)
    
    return models


def predict(model, X):
    """
    Make predictions using the loaded model.
    
    Args:
        model: Trained model object
        X: Feature matrix
        
    Returns:
        predictions: Predicted class labels
        probabilities: Predicted probabilities (if available)
    """
    predictions = model.predict(X)
    
    probabilities = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
    
    return predictions, probabilities


def predict_from_csv(model, csv_path, encoders=None, scaler=None):
    """
    Load data from CSV and make predictions.
    
    Args:
        model: Trained model object
        csv_path: Path to CSV file
        encoders: Pre-fitted encoders (optional)
        scaler: Pre-fitted scaler (optional)
        
    Returns:
        DataFrame with predictions
    """
    df = load_data(csv_path)
    
    # Check if target column exists
    has_target = 'LUNG_CANCER' in df.columns
    
    if has_target:
        X, y, _, _ = preprocess_data(df, fit_encoders=True)
    else:
        # For prediction-only data (no target)
        df_copy = df.copy()
        df_copy['LUNG_CANCER'] = 'NO'  # Dummy target
        X, _, _, _ = preprocess_data(df_copy, fit_encoders=True)
        y = None
    
    predictions, probabilities = predict(model, X)
    
    # Create results DataFrame
    results = df.copy()
    results['PREDICTED'] = ['YES' if p == 1 else 'NO' for p in predictions]
    
    if probabilities is not None:
        results['PROBABILITY_YES'] = probabilities[:, 1]
    
    return results, y


def main():
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(description='Lung Cancer Prediction - Inference')
    parser.add_argument('--model', type=str, default='rf',
                        help='Model to use (ann, svm, xgboost, rf)')
    parser.add_argument('--input', type=str, default=None,
                        help='Input CSV file (default: original dataset)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file for predictions')
    parser.add_argument('--list-models', action='store_true',
                        help='List available saved models')
    args = parser.parse_args()
    
    # List available models
    if args.list_models:
        models = list_available_models()
        if models:
            log.subheader("Available saved models")
            for m in models:
                log.print(f"  - {m}")
        else:
            log.warning("No saved models found. Train models first using:")
            log.print("  python scripts/train.py")
        return
    
    try:
        model = load_saved_model(args.model)
    except FileNotFoundError as e:
        log.error(f"{e}")
        available = list_available_models()
        if available:
            log.info(f"Available models: {', '.join(available)}")
        return
    
    input_file = args.input or DATA_FILE
    log.info(f"Loading data from: {input_file}")
    
    results, y_true = predict_from_csv(model, input_file)
    
    log.header("Prediction Results")
    log.result("Total samples", len(results))
    log.result("Predicted YES", (results['PREDICTED'] == 'YES').sum())
    log.result("Predicted NO", (results['PREDICTED'] == 'NO').sum())
    
    accuracy = None
    if y_true is not None and 'LUNG_CANCER' in results.columns:
        from sklearn.metrics import accuracy_score, classification_report
        y_pred = [1 if p == 'YES' else 0 for p in results['PREDICTED']]
        accuracy = accuracy_score(y_true, y_pred)
        log.result("Accuracy (vs actual)", f"{accuracy:.4f}")
        log.subheader("Classification Report")
        log.print(classification_report(y_true, y_pred, target_names=['NO', 'YES']))
    
    update_prediction_section(
        total_samples=len(results),
        predicted_yes=(results['PREDICTED'] == 'YES').sum(),
        predicted_no=(results['PREDICTED'] == 'NO').sum(),
        accuracy=accuracy
    )
    
    if args.output:
        results.to_csv(args.output, index=False)
        log.success(f"Predictions saved to: {args.output}")
    else:
        log.subheader("Sample predictions (first 10 rows)")
        cols = ['GENDER', 'AGE', 'PREDICTED']
        if 'PROBABILITY_YES' in results.columns:
            cols.append('PROBABILITY_YES')
        if 'LUNG_CANCER' in results.columns:
            cols.insert(-1, 'LUNG_CANCER')
        log.print(results[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
