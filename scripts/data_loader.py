"""
Data loading and preprocessing utilities.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from config import DATA_FILE, RANDOM_SEED
import sys
import os

# Add helpers to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.logger import log


def load_data(filepath=None):
    """
    Load the lung cancer dataset from CSV.
    
    Args:
        filepath: Path to CSV file. If None, uses default from config.
        
    Returns:
        DataFrame with the loaded data.
    """
    if filepath is None:
        filepath = DATA_FILE
    
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df, fit_encoders=True, encoders=None, scaler=None):
    """
    Preprocess the data: encode categorical variables and scale features.
    
    Args:
        df: Input DataFrame
        fit_encoders: If True, fit new encoders. If False, use provided encoders.
        encoders: Dictionary of fitted encoders (used when fit_encoders=False)
        scaler: Fitted StandardScaler (used when fit_encoders=False)
        
    Returns:
        X: Feature matrix (numpy array)
        y: Target vector (numpy array)
        encoders: Dictionary of fitted encoders
        scaler: Fitted StandardScaler
    """
    df = df.copy()
    
    df.columns = df.columns.str.strip()
    
    if encoders is None:
        encoders = {}
    
    if fit_encoders:
        gender_encoder = LabelEncoder()
        df['GENDER'] = gender_encoder.fit_transform(df['GENDER'])
        encoders['GENDER'] = gender_encoder
    else:
        df['GENDER'] = encoders['GENDER'].transform(df['GENDER'])
    
    if fit_encoders:
        target_encoder = LabelEncoder()
        df['LUNG_CANCER'] = target_encoder.fit_transform(df['LUNG_CANCER'])
        encoders['LUNG_CANCER'] = target_encoder
    else:
        df['LUNG_CANCER'] = encoders['LUNG_CANCER'].transform(df['LUNG_CANCER'])
    
    X = df.drop('LUNG_CANCER', axis=1).values
    y = df['LUNG_CANCER'].values
    
    if fit_encoders:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    return X, y, encoders, scaler


def get_feature_names(df=None):
    """
    Get list of feature names (excluding target).
    
    Args:
        df: Input DataFrame. If None, loads from default path.
        
    Returns:
        List of feature names.
    """
    if df is None:
        df = load_data()
    
    df.columns = df.columns.str.strip()
    return [col for col in df.columns if col != 'LUNG_CANCER']


def prepare_data_for_cv(filepath=None):
    """
    Load and preprocess data, ready for cross-validation.
    
    Args:
        filepath: Path to CSV file.
        
    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
    """
    df = load_data(filepath)
    X, y, _, _ = preprocess_data(df)
    feature_names = get_feature_names(df)
    
    return X, y, feature_names


if __name__ == "__main__":
    """
    # Test data loading
    log.info("Loading data...")
    df = load_data()
    log.result("Dataset shape", df.shape)
    log.result("Columns", list(df.columns))
    log.result("Target distribution", f"\n{df['LUNG_CANCER'].value_counts()}")
    
    X, y, feature_names = prepare_data_for_cv()
    log.result("Feature matrix shape", X.shape)
    log.result("Target vector shape", y.shape)
    log.result("Feature names", feature_names)
    """
