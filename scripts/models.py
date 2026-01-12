"""
Model definitions for Lung Cancer Prediction.
Implements: ANN (MLPClassifier), SVM, XGBoost, Random Forest

Note: Using scikit-learn's MLPClassifier instead of TensorFlow/Keras
for better compatibility with Python 3.14+
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from config import ANN_CONFIG, SVM_CONFIG, XGBOOST_CONFIG, RF_CONFIG, RANDOM_SEED

import sys
import os
# Add helpers to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.logger import log


def create_ann_model(input_dim=None, config=None):
    """
    Create ANN classifier using scikit-learn's MLPClassifier.
    
    Architecture: 3 hidden layers with 256, 128, 128 nodes = 512 total (>500 required)
    
    Args:
        input_dim: Number of input features (not needed for MLPClassifier)
        config: Configuration dictionary
        
    Returns:
        MLPClassifier instance
    """
    config = config or ANN_CONFIG
    
    hidden_layer_sizes = tuple(config['layers'])
    
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        max_iter=config.get('epochs', 500),
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False
    )


def create_svm_model(config=None):
    """
    Create SVM classifier.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SVC instance
    """
    config = config or SVM_CONFIG
    return SVC(**config, random_state=RANDOM_SEED)


def create_xgboost_model(config=None):
    """
    Create XGBoost classifier.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        XGBClassifier instance
    """
    config = config or XGBOOST_CONFIG.copy()
    config.pop('use_label_encoder', None)
    return XGBClassifier(**config)


def create_rf_model(config=None):
    """
    Create Random Forest classifier.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        RandomForestClassifier instance
    """
    config = config or RF_CONFIG
    return RandomForestClassifier(**config)


def get_model_by_name(name, input_dim=None):
    """
    Get model by name.
    
    Args:
        name: Model name ('ann', 'svm', 'xgboost', 'rf')
        input_dim: Number of input features (optional, for compatibility)
        
    Returns:
        Model instance
    """
    name = name.lower()
    
    if name == 'ann':
        return create_ann_model(input_dim)
    elif name == 'svm':
        return create_svm_model()
    elif name == 'xgboost':
        return create_xgboost_model()
    elif name == 'rf':
        return create_rf_model()
    else:
        raise ValueError(f"Unknown model: {name}")


if __name__ == "__main__":
    """
    # Test model creation
    log.info("Testing model creation...")
    
    # Create dummy data
    X = np.random.randn(100, 15)
    y = np.random.randint(0, 2, 100)
    
    log.subheader("1. ANN Model (MLPClassifier)")
    ann = create_ann_model(input_dim=15)
    log.result("Hidden layers", ann.hidden_layer_sizes)
    total_nodes = sum(ann.hidden_layer_sizes)
    log.result("Total hidden nodes", f"{total_nodes} (requirement: >= 500)")
    
    log.subheader("2. SVM Model")
    svm = create_svm_model()
    log.result("SVM config", SVM_CONFIG)
    
    log.subheader("3. XGBoost Model")
    xgb = create_xgboost_model()
    log.result("XGBoost n_estimators", xgb.n_estimators)
    
    log.subheader("4. Random Forest Model")
    rf = create_rf_model()
    log.result("RF n_estimators", rf.n_estimators)
    
    log.success("All models created successfully!")
    """
