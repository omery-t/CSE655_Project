"""
Missing value injection and imputation utilities.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

from config import MISSING_VALUE_CONFIG, RANDOM_SEED


def inject_missing_values(X, percentage=None, n_features_to_null=None, random_state=None):
    """
    Inject synthetic missing values into the data.
    
    Randomly selects a percentage of rows and sets one feature to NaN per row.
    
    Args:
        X: Feature matrix (numpy array)
        percentage: Percentage of rows to affect (0-1)
        n_features_to_null: Number of features to set NULL per row
        random_state: Random seed for reproducibility
        
    Returns:
        X_missing: Feature matrix with missing values
        missing_mask: Boolean mask indicating missing positions
    """
    if percentage is None:
        percentage = MISSING_VALUE_CONFIG['percentage']
    if n_features_to_null is None:
        n_features_to_null = MISSING_VALUE_CONFIG['n_features_to_null']
    if random_state is None:
        random_state = RANDOM_SEED
    
    np.random.seed(random_state)
    
    X_missing = X.copy().astype(float)
    n_samples, n_features = X_missing.shape
    
    # Calculate number of rows to affect
    n_rows_to_affect = int(np.ceil(n_samples * percentage))
    
    # Randomly select rows
    rows_to_affect = np.random.choice(n_samples, size=n_rows_to_affect, replace=False)
    
    # Create missing mask
    missing_mask = np.zeros_like(X_missing, dtype=bool)
    
    for row_idx in rows_to_affect:
        # Randomly select feature(s) to set as NULL
        features_to_null = np.random.choice(
            n_features, 
            size=min(n_features_to_null, n_features), 
            replace=False
        )
        for feature_idx in features_to_null:
            X_missing[row_idx, feature_idx] = np.nan
            missing_mask[row_idx, feature_idx] = True
    
    return X_missing, missing_mask


def impute_missing_values(X_missing, method='knn', n_neighbors=5):
    """
    Impute missing values using specified method.
    
    Args:
        X_missing: Feature matrix with missing values
        method: Imputation method ('knn', 'mean', 'median')
        n_neighbors: Number of neighbors for KNN imputation
        
    Returns:
        X_imputed: Feature matrix with imputed values
        imputer: Fitted imputer object
    """
    if method == 'knn':
        imputer = KNNImputer(n_neighbors=n_neighbors)
    else:
        imputer = SimpleImputer(strategy=method)
    
    X_imputed = imputer.fit_transform(X_missing)
    
    return X_imputed, imputer


def get_missing_value_summary(X_missing):
    """
    Get summary of missing values.
    
    Args:
        X_missing: Feature matrix with missing values
        
    Returns:
        Dictionary with missing value statistics
    """
    n_missing = np.isnan(X_missing).sum()
    n_total = X_missing.size
    n_rows_affected = np.any(np.isnan(X_missing), axis=1).sum()
    
    return {
        'total_missing': n_missing,
        'total_elements': n_total,
        'missing_percentage': (n_missing / n_total) * 100,
        'rows_affected': n_rows_affected,
        'rows_affected_percentage': (n_rows_affected / X_missing.shape[0]) * 100
    }


if __name__ == "__main__":
    """
    import sys
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, BASE_DIR)
    from helpers.logger import log
    # Test missing value injection and imputation
    log.info("Testing missing value utilities...")
    
    # Create dummy data with missing values
    X = np.random.randn(10, 5)
    X[0, 0] = np.nan
    X[2, 3] = np.nan
    X[5, 1] = np.nan
    
    log.result("Original data shape", X.shape)
    log.result("Original missing values", np.isnan(X).sum())
    
    # Test injection
    X, _ = inject_missing_values(X, 0.1)
    summary = get_missing_value_summary(X)
    
    log.subheader("After injection")
    log.result("Total missing", summary['total_missing'])
    log.result("Missing percentage", f"{summary['missing_percentage']:.2f}%")
    log.result("Rows affected", summary['rows_affected'])
    log.result("Rows affected percentage", f"{summary['rows_affected_percentage']:.2f}%")
    
    # Test imputation
    X_imputed, _ = impute_missing_values(X)
    
    log.subheader("After imputation")
    log.result("Missing values", np.isnan(X_imputed).sum())
    log.result("Data shape", X_imputed.shape)
    """
