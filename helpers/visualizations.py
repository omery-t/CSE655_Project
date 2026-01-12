"""
Visualization utilities for generating detailed plots.
Generates: feature_importance.png, confusion_matrix.png, roc_curve.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Get project paths
HELPERS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(HELPERS_DIR)
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

import sys
# Add helpers to path
sys.path.insert(0, BASE_DIR)
from helpers.logger import log


def ensure_reports_dir():
    """Ensure reports directory exists."""
    os.makedirs(REPORTS_DIR, exist_ok=True)


def plot_feature_importance(model, feature_names, model_name='model', save_path=None):
    """
    Generate and save feature importance plot.
    
    Args:
        model: Trained model (must have feature_importances_ or coef_)
        feature_names: List of feature names
        model_name: Name of the model for title
        save_path: Path to save the plot
        
    Returns:
        Path to saved plot or None if not applicable
    """
    ensure_reports_dir()
    
    if save_path is None:
        save_path = os.path.join(REPORTS_DIR, 'feature_importance.png')
    
    importances = None
    
    # Try to get feature importances (tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    # Try to get coefficients (linear models, SVM)
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
        if len(importances) != len(feature_names):
            importances = np.abs(model.coef_[0])
    # For MLPClassifier - use first layer weights
    elif hasattr(model, 'coefs_') and len(model.coefs_) > 0:
        importances = np.abs(model.coefs_[0]).sum(axis=1)
    
    if importances is None:
        log.warning(f"Feature importance not available for {model_name}")
        return None
    
    # Ensure correct length
    if len(importances) != len(feature_names):
        log.error(f"Feature count mismatch for {model_name}")
        return None
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Create plot
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("viridis", len(sorted_names))
    
    bars = plt.barh(range(len(sorted_names)), sorted_importances, color=colors)
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Feature Importance - {model_name.upper()}', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    log.success(f"Saved: {save_path}")
    return save_path


def plot_confusion_matrix(y_true, y_pred, class_names=['NO', 'YES'], save_path=None):
    """
    Generate and save confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save the plot
        
    Returns:
        Path to saved plot
    """
    ensure_reports_dir()
    
    if save_path is None:
        save_path = os.path.join(REPORTS_DIR, 'confusion_matrix.png')
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 16})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    log.success(f"Saved: {save_path}")
    return save_path


def plot_roc_curve(y_true, y_proba, save_path=None):
    """
    Generate and save ROC curve plot.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        save_path: Path to save the plot
        
    Returns:
        Path to saved plot
    """
    ensure_reports_dir()
    
    if save_path is None:
        save_path = os.path.join(REPORTS_DIR, 'roc_curve.png')
    
    # Probability array format
    if len(y_proba.shape) > 1:
        y_proba = y_proba[:, 1]
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#2ecc71', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#e74c3c', lw=2, linestyle='--', label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.3, color='#2ecc71')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    log.success(f"Saved: {save_path}")
    return save_path


def generate_all_plots(model, y_true, y_pred, y_proba, feature_names, model_name='model'):
    """
    Generate all visualization plots at once.
    
    Args:
        model: Trained model
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        feature_names: List of feature names
        model_name: Name of the model
        
    Returns:
        Dictionary of generated plot paths
    """
    log.info("Generating visualization plots...")
    
    paths = {}
    
    # Feature importance
    fi_path = plot_feature_importance(model, feature_names, model_name)
    if fi_path:
        paths['feature_importance'] = fi_path
    
    # Confusion matrix
    cm_path = plot_confusion_matrix(y_true, y_pred)
    paths['confusion_matrix'] = cm_path
    
    # ROC curve
    if y_proba is not None:
        roc_path = plot_roc_curve(y_true, y_proba)
        paths['roc_curve'] = roc_path
    
    return paths

# Dummy test when run directly, delete the plot files afterwards
if __name__ == "__main__":
    """
    # Dummy data
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 1, 1, 1, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 1, 1, 0, 1, 0, 1])
    y_proba = np.random.rand(10, 2)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    
    feature_names = ['AGE', 'SMOKING', 'ANXIETY', 'FATIGUE', 'COUGHING']
    
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_proba)
    """
