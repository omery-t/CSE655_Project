"""
Configuration settings for the Lung Cancer Prediction project.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_FILE = os.path.join(DATA_DIR, 'survey lung cancer.csv')

# Random seed for reproducibility
RANDOM_SEED = 42

# Cross-validation settings
N_FOLDS = 5

# ANN Hyperparameters (using scikit-learn MLPClassifier)
ANN_CONFIG = {
    'layers': [256, 128, 128],  # Hidden layer sizes (total 512 nodes > 500 required)
    'epochs': 500,  # max_iter for MLPClassifier
}

# SVM Hyperparameters
SVM_CONFIG = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
    'probability': True
}

# XGBoost Hyperparameters
XGBOOST_CONFIG = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': RANDOM_SEED
}

# Random Forest Hyperparameters
RF_CONFIG = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': RANDOM_SEED
}

# Missing Value Configuration
MISSING_VALUE_CONFIG = {
    'percentage': 0.10,  # 10% of training rows
    'n_features_to_null': 1  # Number of features to set NULL per row
}
