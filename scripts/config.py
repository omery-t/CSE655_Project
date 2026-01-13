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

# ANN Hyperparameters (using custom PyTorch architecture)
ANN_CONFIG = {
    'layers': [16, 8],     # "Minimalist" architecture to prevent overfitting
    'dropout_rate': 0.3, 
    'learning_rate': 0.0005,
    'weight_decay': 1e-2,  # Strong L2 regularization
    'batch_size': 16,      # Smaller batch for more gradients updates
    'epochs': 500,
    'random_state': RANDOM_SEED,
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

# Logistic Regression Hyperparameters
LR_CONFIG = {
    'max_iter': 1000,
    'random_state': RANDOM_SEED,
    'solver': 'lbfgs'
}

# KNN Hyperparameters
KNN_CONFIG = {
    'n_neighbors': 5,
    'weights': 'uniform',
    'metric': 'minkowski'
}

# Naive Bayes Hyperparameters
NB_CONFIG = {}  # GaussianNB doesn't have many hyperparameters to tune

# Extra Trees Hyperparameters
ET_CONFIG = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'random_state': RANDOM_SEED
}

# Ensemble Configuration
ENSEMBLE_CONFIG = {
    'voting': 'soft',  # 'hard' or 'soft'
}
