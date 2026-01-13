"""
Report Manager for modular master_report.txt updates.
Updates sections independently with timestamps.
"""

import os
from datetime import datetime

# Get project paths
HELPERS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(HELPERS_DIR)
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
REPORT_FILE = os.path.join(REPORTS_DIR, 'master_report.txt')

import sys
# Add helpers to path
sys.path.insert(0, BASE_DIR)
from helpers.logger import log

# Section markers
SECTIONS = {
    'HEADER': '=' * 70,
    'TRAINING': '[FOLD RESULTS]',
    'PREDICTION': '[PREDICTION]',
    'MISSING_VALUES': '[SYNTHETIC RESULTS]',
    'SYSTEM_INFO': '[SYSTEM INFO]'
}


def ensure_reports_dir():
    """Ensure reports directory exists."""
    os.makedirs(REPORTS_DIR, exist_ok=True)


def get_timestamp():
    """Get formatted timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def initialize_report():
    """
    Initialize a fresh master report with header.
    
    Returns:
        Path to report file
    """
    ensure_reports_dir()
    
    header = f"""{'=' * 70}
              LUNG CANCER PREDICTION - MASTER REPORT
{'=' * 70}
Report Generated: {get_timestamp()}
Project: Lung Cancer Model Comparison

This report is automatically updated after each operation.
{'=' * 70}

"""
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(header)
    
    log.success(f"Initialized: {REPORT_FILE}")
    return REPORT_FILE


def read_report():
    """
    Read current report contents.
    
    Returns:
        Report content as string, or None if not exists
    """
    if not os.path.exists(REPORT_FILE):
        return None
    
    with open(REPORT_FILE, 'r', encoding='utf-8') as f:
        return f.read()


def find_section(content, section_marker):
    """
    Find the start and end indices of a section.
    
    Args:
        content: Full report content
        section_marker: Section marker string (e.g., '[TRAINING]')
        
    Returns:
        Tuple of (start_idx, end_idx) or (None, None) if not found
    """
    start = content.find(section_marker)
    if start == -1:
        return None, None
    
    # Find end of section (next section marker or end of file)
    end = len(content)
    for marker in SECTIONS.values():
        if marker == section_marker or marker == SECTIONS['HEADER']:
            continue
        idx = content.find(marker, start + len(section_marker))
        if idx != -1 and idx < end:
            end = idx
    
    return start, end


def update_section(section_name, content_lines):
    """
    Update a specific section of the master report.
    
    Args:
        section_name: Name of section ('TRAINING', 'PREDICTION', 'MISSING_VALUES', 'SYSTEM_INFO')
        content_lines: List of lines to add to the section
        
    Returns:
        Path to updated report
    """
    ensure_reports_dir()
    
    # Get section marker
    section_marker = SECTIONS.get(section_name.upper())
    if not section_marker:
        raise ValueError(f"Unknown section: {section_name}")
    
    # Initialize if not exists
    if not os.path.exists(REPORT_FILE):
        initialize_report()
    
    current = read_report()
    
    # Format new section content
    timestamp = get_timestamp()
    new_section = f"\n{section_marker}\n"
    new_section += f"Last Updated: {timestamp}\n"
    new_section += "-" * 50 + "\n"
    
    for line in content_lines:
        new_section += f"{line}\n"
    
    new_section += "\n"
    
    # Check if section exists
    start, end = find_section(current, section_marker)
    
    if start is not None:
        # Replace existing section
        updated = current[:start] + new_section + current[end:]
    else:
        # Append new section
        updated = current + new_section
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(updated)
    
    log.success(f"Updated section: {section_name}")
    return REPORT_FILE


def update_training_section(results, best_model):
    """
    Update the TRAINING section with model comparison results.
    
    Args:
        results: Dictionary with model results
        best_model: Name of best performing model
    """
    lines = [
        f"Best Model: {best_model.upper()}",
        "",
        "Model Performance (5-Fold Cross-Validation):",
        "-" * 45,
        f"{'Model':<12} {'Accuracy':<12} {'F1-Score':<12} {'AUC-ROC':<12}",
    ]
    
    for model_name, metrics in results.items():
        acc = metrics['mean']['accuracy']
        f1 = metrics['mean']['f1']
        auc_val = metrics['mean'].get('auc_roc', 0)
        lines.append(f"{model_name:<12} {acc:.4f}       {f1:.4f}       {auc_val:.4f}")
    
    update_section('TRAINING', lines)


def update_prediction_section(total_samples, predicted_yes, predicted_no, accuracy=None):
    """
    Update the PREDICTION section with inference results.
    
    Args:
        total_samples: Total number of samples predicted
        predicted_yes: Count of YES predictions
        predicted_no: Count of NO predictions
        accuracy: Optional accuracy if ground truth available
    """
    lines = [
        f"Total Samples: {total_samples}",
        f"Predicted YES: {predicted_yes}",
        f"Predicted NO:  {predicted_no}",
    ]
    
    if accuracy is not None:
        lines.append(f"Accuracy: {accuracy:.4f}")
    
    update_section('PREDICTION', lines)


def update_missing_values_section(rows_affected, percentage, imputation_method, results=None, best_model=None):
    """
    Update the MISSING VALUES section with imputation stats and model results.
    
    Args:
        rows_affected: Number of rows with missing values
        percentage: Percentage of rows affected
        imputation_method: Method used for imputation
        results: Optional dictionary with model results
        best_model: Optional name of best performing model
    """
    lines = [
        f"Rows Affected: {rows_affected} ({percentage:.1f}%)",
        f"Imputation Method: {imputation_method}",
        "Status: Successfully imputed",
    ]
    
    if results and best_model:
        lines.append("")
        lines.append(f"Best Model (Synthetic): {best_model.upper()}")
        lines.append("")
        lines.append("Model Performance (After Imputation):")
        lines.append("-" * 45)
        lines.append(f"{'Model':<12} {'Accuracy':<12} {'F1-Score':<12} {'AUC-ROC':<12}")
        
        for model_name, metrics in results.items():
            acc = metrics['mean']['accuracy']
            f1 = metrics['mean']['f1']
            auc_val = metrics['mean'].get('auc_roc', 0)
            lines.append(f"{model_name:<12} {acc:.4f}       {f1:.4f}       {auc_val:.4f}")
    
    update_section('MISSING_VALUES', lines)


def reset_report():
    """Reset the master report to initial state."""
    if os.path.exists(REPORT_FILE):
        os.remove(REPORT_FILE)
    return initialize_report()


def view_report():
    """
    Display the current master report.
    
    Returns:
        Report content as string
    """
    if not os.path.exists(REPORT_FILE):
        log.warning("No report found. Run training or prediction first.")
        return None
    
    content = read_report()
    log.print(content)
    return content


if __name__ == "__main__":
    """
    # Test report manager
    log.info("Testing report manager...")
    
    # Initialize
    reset_report()
    
    # Test training update
    test_results = {
        'ann': {'mean': {'accuracy': 0.87, 'f1': 0.93, 'auc_roc': 0.58}},
        'svm': {'mean': {'accuracy': 0.90, 'f1': 0.94, 'auc_roc': 0.92}},
        'rf': {'mean': {'accuracy': 0.91, 'f1': 0.95, 'auc_roc': 0.93}},
    }
    update_training_section(test_results, 'rf')
    
    # Test prediction update
    update_prediction_section(100, 85, 15, 0.95)
    
    # Test missing values update
    update_missing_values_section(10, 10.0, 'KNN (k=5)')
    
    # View report
    log.section("Viewing Report")
    view_report()
    """
