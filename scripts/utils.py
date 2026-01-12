"""
Utility functions for visualization and reporting.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

import os
from config import BASE_DIR
import sys
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from helpers.logger import log


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for AUC-ROC)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
        except:
            metrics['auc_roc'] = np.nan
    
    return metrics


def print_results_table(results, title="Model Comparison Results"):
    """
    Print results as a formatted table.
    
    Args:
        results: Dictionary with model names as keys and metrics dict as values
        title: Title for the table
    """
    log.header(title)
    
    # Header
    log.print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC-ROC':<12}")
    log.print("-" * 80)
    
    for model_name, metrics in results.items():
        if 'mean' in metrics:
            # Cross-validation results
            acc = f"{metrics['mean']['accuracy']:.4f} +/- {metrics['std']['accuracy']:.4f}"
            prec = f"{metrics['mean']['precision']:.4f}"
            rec = f"{metrics['mean']['recall']:.4f}"
            f1 = f"{metrics['mean']['f1']:.4f}"
            auc = f"{metrics['mean'].get('auc_roc', np.nan):.4f}" if not np.isnan(metrics['mean'].get('auc_roc', np.nan)) else "N/A"
        else:
            acc = f"{metrics['accuracy']:.4f}"
            prec = f"{metrics['precision']:.4f}"
            rec = f"{metrics['recall']:.4f}"
            f1 = f"{metrics['f1']:.4f}"
            auc = f"{metrics.get('auc_roc', np.nan):.4f}" if not np.isnan(metrics.get('auc_roc', np.nan)) else "N/A"
        
        log.print(f"{model_name:<15} {acc:<12} {prec:<12} {rec:<12} {f1:<12} {auc:<12}")
    
    log.print("=" * 80)


def plot_model_comparison(results, metric='accuracy', save_path=None):
    """
    Create bar plot comparing models.
    
    Args:
        results: Dictionary with model results
        metric: Metric to plot
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    models = list(results.keys())
    means = [results[m]['mean'][metric] for m in models]
    stds = [results[m]['std'][metric] for m in models]
    
    colors = sns.color_palette("husl", len(models))
    bars = plt.bar(models, means, yerr=stds, capsize=5, color=colors, edgecolor='black')
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}', fontsize=14)
    plt.ylim(0, 1.1)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.success(f"Plot saved to: {save_path}")
    
    plt.close()


def plot_all_metrics(results, save_path=None):
    """
    Create grouped bar plot for all metrics.
    
    Args:
        results: Dictionary with model results
        save_path: Path to save the figure
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    models = list(results.keys())
    
    x = np.arange(len(metrics))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("husl", len(models))
    
    for i, model in enumerate(models):
        values = [results[model]['mean'][m] for m in metrics]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model, color=colors[i])
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.success(f"Plot saved to: {save_path}")
    
    plt.close()


def plot_cv_boxplot(cv_scores, metric='accuracy', save_path=None):
    """
    Create boxplot of cross-validation scores.
    
    Args:
        cv_scores: Dictionary with model names and list of fold scores
        metric: Metric being plotted
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    data = []
    labels = []
    for model, scores in cv_scores.items():
        data.append(scores)
        labels.append(model)
    
    bp = plt.boxplot(data, labels=labels, patch_artist=True)
    
    colors = sns.color_palette("husl", len(labels))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(f'Cross-Validation Scores - {metric.replace("_", " ").title()}', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.success(f"Plot saved to: {save_path}")
    
    plt.close()


def save_results_to_file(results, filename, title="Results"):
    """
    Save results to a text file.
    
    Args:
        results: Dictionary with model results
        filename: Output filename
        title: Title for the results
    """
    filepath = os.path.join(BASE_DIR, filename)
    
    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f" {title}\n")
        f.write("=" * 80 + "\n\n")
        
        for model_name, metrics in results.items():
            f.write(f"\n{model_name}:\n")
            f.write("-" * 40 + "\n")
            
            if 'mean' in metrics:
                for metric, value in metrics['mean'].items():
                    std = metrics['std'].get(metric, 0)
                    f.write(f"  {metric}: {value:.4f} +/- {std:.4f}\n")
            else:
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    log.success(f"Results saved to: {filepath}")
    return filepath
