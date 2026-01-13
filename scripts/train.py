"""
Main training script for Lung Cancer Prediction.
Implements 5-fold cross-validation for model comparison.

Usage:
    python train.py                    # Run Fold Comparison (original data)
    python train.py --missing-values   # Run Synthetic Comparison (with missing values)
    python train.py --all              # Run both comparisons
"""

import os
import sys
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
import joblib

from config import N_FOLDS, RANDOM_SEED, MODELS_DIR, BASE_DIR
from data_loader import prepare_data_for_cv, load_data, preprocess_data, get_feature_names
from models import get_model_by_name
from utils import (
    calculate_metrics, print_results_table, 
    plot_model_comparison, plot_all_metrics, plot_cv_boxplot,
    save_results_to_file, plot_training_history
)
from missing_values import inject_missing_values, impute_missing_values, get_missing_value_summary

sys.path.insert(0, os.path.dirname(BASE_DIR))
from helpers.visualizations import generate_all_plots, plot_feature_importance, plot_confusion_matrix, plot_roc_curve
from helpers.report_manager import update_training_section, update_missing_values_section
from helpers.logger import log


def run_cross_validation(X, y, model_names=['ann', 'svm', 'xgboost', 'rf', 'lr', 'knn', 'nb', 'et', 'voting', 'stacking'], n_folds=N_FOLDS):
    """
    Run k-fold cross-validation for all models.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_names: List of model names to evaluate
        n_folds: Number of cross-validation folds
        
    Returns:
        results: Dictionary with mean and std metrics for each model
        cv_scores: Dictionary with per-fold scores for each model
    """
    log.header(f"Running {n_folds}-Fold Cross-Validation")
    log.result("Dataset size", f"{X.shape[0]} samples, {X.shape[1]} features")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    results = {}
    cv_scores = {name: {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc_roc': []} 
                 for name in model_names}
    cv_predictions = {name: {'y_true': [], 'y_pred': [], 'y_proba': []} for name in model_names}
    
    for model_name in model_names:
        log.subheader(f"[{model_name.upper()}] Training...")
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = get_model_by_name(model_name, input_dim=X.shape[1])
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_val)
            else:
                y_proba = None
            
            # Store CV predictions
            cv_predictions[model_name]['y_true'].append(y_val)
            cv_predictions[model_name]['y_pred'].append(y_pred)
            if y_proba is not None:
                cv_predictions[model_name]['y_proba'].append(y_proba)
            
            metrics = calculate_metrics(y_val, y_pred, y_proba)
            fold_metrics.append(metrics)
            
            for metric, value in metrics.items():
                cv_scores[model_name][metric].append(value)
            
            log.print(f"  Fold {fold+1}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
        
        # Concatenate CV predictions
        cv_predictions[model_name]['y_true'] = np.concatenate(cv_predictions[model_name]['y_true'])
        cv_predictions[model_name]['y_pred'] = np.concatenate(cv_predictions[model_name]['y_pred'])
        if cv_predictions[model_name]['y_proba']:
            cv_predictions[model_name]['y_proba'] = np.concatenate(cv_predictions[model_name]['y_proba'])
        else:
            cv_predictions[model_name]['y_proba'] = None

        results[model_name] = {
            'mean': {},
            'std': {}
        }
        
        for metric in fold_metrics[0].keys():
            values = [m[metric] for m in fold_metrics]
            results[model_name]['mean'][metric] = np.nanmean(values)
            results[model_name]['std'][metric] = np.nanstd(values)
        
        log.result("Mean Accuracy", f"{results[model_name]['mean']['accuracy']:.4f} +/- {results[model_name]['std']['accuracy']:.4f}")
    
    return results, cv_scores, cv_predictions


def save_best_model(X, y, model_name, output_dir=None, prefix=''):
    """
    Train and save the best model on full dataset.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_name: Name of the model to save
        output_dir: Directory to save model
        prefix: Optional prefix for the filename
    """
    if output_dir is None:
        output_dir = MODELS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    log.info(f"Saving best model ({model_name}) with prefix '{prefix}'...")
    
    model = get_model_by_name(model_name, input_dim=X.shape[1])
    model.fit(X, y)
    
    filename = f'{prefix}{model_name}_model.joblib' if prefix else f'{model_name}_model.joblib'
    model_path = os.path.join(output_dir, filename)
    joblib.dump(model, model_path)
    
    log.success(f"Model saved to: {model_path}")
    return model_path, model


def run_fold(X, y, feature_names):
    """
    Fold: Model comparison with original data.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        
    Returns:
        results: Dictionary with model results
        cv_scores: Dictionary with per-fold scores
    """
    log.header("FOLD COMPARISON: Model Comparison (Original Data)")
    
    results, cv_scores, cv_predictions = run_cross_validation(X, y)
    
    print_results_table(results, "Fold Results: Model Comparison Results")
    
    best_model = max(results.keys(), key=lambda k: results[k]['mean']['accuracy'])
    log.success(f"Best Model: {best_model.upper()} (Accuracy: {results[best_model]['mean']['accuracy']:.4f})")
    
    plots_dir = os.path.join(BASE_DIR, 'results')
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_all_metrics(results, save_path=os.path.join(plots_dir, 'fold_metrics_comparison.png'))
    plot_cv_boxplot({k: v['accuracy'] for k, v in cv_scores.items()}, 
                    metric='accuracy',
                    save_path=os.path.join(plots_dir, 'fold_cv_boxplot.png'))
    
    save_results_to_file(results, 'results/fold_results.txt', 
                        "Fold Comparison (Original Data)")
    
    # Save best model trained on full data (for deployment/feature importance)
    model_path, best_model_instance = save_best_model(X, y, best_model)
    
    # Use CV predictions for plotting (evaluates generalization)
    y_true_cv = cv_predictions[best_model]['y_true']
    y_pred_cv = cv_predictions[best_model]['y_pred']
    y_proba_cv = cv_predictions[best_model]['y_proba']
    
    generate_all_plots(best_model_instance, y_true_cv, y_pred_cv, y_proba_cv, feature_names, best_model)
    
    # Plot ANN training history if available (using best model if it's ANN or just retrain once)
    log.info("Generating ANN training history plot...")
    ann_model = get_model_by_name('ann', input_dim=X.shape[1])
    ann_model.fit(X, y)
    plot_training_history(ann_model.history, save_path=os.path.join(plots_dir, 'ann_training_history.png'))
    
    update_training_section(results, best_model)
    
    return results, cv_scores, best_model


def run_synthetic(X, y, feature_names):
    """
    Synthetic: Model comparison with synthetic missing values.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        
    Returns:
        results: Dictionary with model results
        cv_scores: Dictionary with per-fold scores
    """
    log.header("SYNTHETIC COMPARISON: Model Comparison (With Missing Values)")
    
    log.info("Injecting synthetic missing values...")
    X_missing, mask = inject_missing_values(X, percentage=0.10)
    summary = get_missing_value_summary(X_missing)
    
    log.result("Rows affected", f"{summary['rows_affected']} ({summary['rows_affected_percentage']:.1f}%)")
    log.result("Total missing values", summary['total_missing'])
    
    log.info("Imputing missing values using KNN Imputer (k=5)...")
    X_imputed, imputer = impute_missing_values(X_missing, method='knn', n_neighbors=5)
    log.result("Imputation complete", f"Missing values after: {np.isnan(X_imputed).sum()}")
    
    results, cv_scores, _ = run_cross_validation(X_imputed, y)
    
    print_results_table(results, "Synthetic Comparison Results (After Imputation)")
    
    best_model = max(results.keys(), key=lambda k: results[k]['mean']['accuracy'])
    log.success(f"Best Model: {best_model.upper()} (Accuracy: {results[best_model]['mean']['accuracy']:.4f})")
    
    plots_dir = os.path.join(BASE_DIR, 'results')
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_all_metrics(results, save_path=os.path.join(plots_dir, 'synthetic_metrics_comparison.png'))
    plot_cv_boxplot({k: v['accuracy'] for k, v in cv_scores.items()}, 
                    metric='accuracy',
                    save_path=os.path.join(plots_dir, 'synthetic_cv_boxplot.png'))
    
    save_results_to_file(results, 'results/synthetic_results.txt', 
                        "Synthetic Comparison (With Missing Values)")
    
    # Save best synthetic model
    save_best_model(X_imputed, y, best_model, prefix='synthetic_')
    
    update_missing_values_section(
        summary['rows_affected'],
        summary['rows_affected_percentage'],
        'KNN Imputer (k=5)',
        results=results,
        best_model=best_model
    )
    
    return results, cv_scores, best_model


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Lung Cancer Prediction Model Training')
    parser.add_argument('--missing-values', action='store_true', 
                        help='Run Synthetic Comparison with missing values')
    parser.add_argument('--all', action='store_true',
                        help='Run both Fold and Synthetic Comparisons')
    args = parser.parse_args()
    
    # Load and preprocess data
    log.info("Loading data...")
    X, y, feature_names = prepare_data_for_cv()
    log.result("Data loaded", f"{X.shape[0]} samples, {X.shape[1]} features")
    log.result("Class distribution", f"{np.bincount(y.astype(int))}")
    
    if args.all:
        log.header("RUNNING BOTH COMPARISONS")
        
        results_fold, cv_fold, best_fold = run_fold(X, y, feature_names)
        results_syn, cv_syn, best_syn = run_synthetic(X, y, feature_names)
        
        log.header("SUMMARY: Fold vs Synthetic Comparison")
        
        log.print(f"\n{'Model':<15} {'Fold Acc':<15} {'Synthetic Acc':<15} {'Difference':<15}")
        log.print("-" * 60)
        
        for model in results_fold.keys():
            acc1 = results_fold[model]['mean']['accuracy']
            acc2 = results_syn[model]['mean']['accuracy']
            diff = acc2 - acc1
            sign = '+' if diff >= 0 else ''
            log.print(f"{model:<15} {acc1:.4f}{'':<10} {acc2:.4f}{'':<10} {sign}{diff:.4f}")
        
        log.print("-" * 60)
        log.print(f"Best Fold: {best_fold.upper()}")
        log.print(f"Best Synthetic: {best_syn.upper()}")
        
    elif args.missing_values:
        # Run only Synthetic
        run_synthetic(X, y, feature_names)
    else:
        # Run only Fold
        run_fold(X, y, feature_names)
    
    log.header("Training Complete!")
    log.success(f"Results saved to: {os.path.join(BASE_DIR, 'results')}")
    log.success(f"Models saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
