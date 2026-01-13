"""
Model definitions for Lung Cancer Prediction.
Implements: ANN (MLPClassifier), SVM, XGBoost, Random Forest

Note: Using custom PyTorch implementation for the ANN
to allow explicit control over layers, loss, and optimizers.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import (
    ANN_CONFIG, SVM_CONFIG, XGBOOST_CONFIG, RF_CONFIG, 
    LR_CONFIG, KNN_CONFIG, NB_CONFIG, ET_CONFIG, ENSEMBLE_CONFIG,
    RANDOM_SEED
)

import sys
import os
# Add helpers to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.logger import log


class LungCancerANN(nn.Module):
    """
    Custom PyTorch Neural Network architecture for Lung Cancer Prediction.
    
    Architecture:
    Input (15) -> Dense(128) + BatchNorm + ReLU -> Dropout(0.3) -> Dense(64) + ReLU -> Dense(1)
    (Note: Sigmoid removed to use BCEWithLogitsLoss for better stability)
    """
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout_rate=0.3):
        super(LungCancerANN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.output = nn.Linear(hidden_dims[1], 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.output(x)
        return x


class SklearnTorchWrapper(ClassifierMixin, BaseEstimator):
    """
    Scikit-learn compatible wrapper for the PyTorch ANN model.
    Allows it to be used in cross-validation and pipelines.
    """
    _estimator_type = "classifier"
    
    def __init__(self, input_dim=15, config=None):
        self.input_dim = input_dim
        self.config = config

    def fit(self, X, y):
        # Set seed for reproducibility
        torch.manual_seed(RANDOM_SEED)
        
        # Determine classes
        self.classes_ = np.unique(y)
        
        # Config handling
        config = self.config or ANN_CONFIG
        
        # Split for internal validation (for early stopping)
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
        )
        
        # Apply SMOTE to the training set only
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=RANDOM_SEED)
        try:
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            log.info(f"SMOTE applied: {X_train.shape[0]} -> {X_train_res.shape[0]} samples")
        except Exception as e:
            log.warning(f"SMOTE failed: {e}. Using original training data.")
            X_train_res, y_train_res = X_train, y_train
        
        # Calculate pos_weight (should be 1.0 if SMOTE worked perfectly, but let's be safe)
        num_pos = np.sum(y_train_res == 1)
        num_neg = np.sum(y_train_res == 0)
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)
        
        input_dim = X_train_res.shape[1]
        self.model = LungCancerANN(
            input_dim=input_dim, 
            hidden_dims=config['layers'],
            dropout_rate=config.get('dropout_rate', 0.4)
        )
        
        # Convert to tensors
        X_train_t = torch.tensor(X_train_res, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_res, dtype=torch.float32).view(-1, 1)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        
        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=config.get('batch_size', 32), shuffle=True)
        
        # Use BCEWithLogitsLoss with pos_weight
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0)
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        self.history = {'loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 25
        best_state = None

        for epoch in range(config.get('epochs', 200)):
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(loader)
            self.history['loss'].append(avg_train_loss)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = nn.BCEWithLogitsLoss()(val_outputs, y_val_t).item()
                self.history['val_loss'].append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early Stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                # log.info(f"Early stopping at epoch {epoch}")
                break
        
        if best_state:
            self.model.load_state_dict(best_state)
            
        # Optimize threshold on validation set
        self.model.eval()
        with torch.no_grad():
            val_logits = self.model(X_val_t)
            val_probs = torch.sigmoid(val_logits).numpy().flatten()
            
            best_f1 = -1
            self.threshold = 0.5
            for t in np.linspace(0.1, 0.9, 81):
                preds = (val_probs > t).astype(int)
                from sklearn.metrics import f1_score
                f1 = f1_score(y_val, preds)
                if f1 > best_f1:
                    best_f1 = f1
                    self.threshold = t
        
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).numpy().flatten()
            predictions = (probs > self.threshold).astype(int)
        return predictions

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.model(X_tensor)
            probs_yes = torch.sigmoid(logits).numpy()
            probs_no = 1 - probs_yes
            probabilities = np.hstack([probs_no, probs_yes])
        return probabilities


def create_ann_model(input_dim=15, config=None):
    """
    Create ANN classifier using the custom PyTorch wrapper.
    """
    return SklearnTorchWrapper(input_dim=input_dim, config=config)


def create_svm_model(config=None):
    """Create SVM classifier."""
    config = config or SVM_CONFIG
    return SVC(**config, random_state=RANDOM_SEED)


def create_xgboost_model(config=None):
    """Create XGBoost classifier."""
    config = config or XGBOOST_CONFIG.copy()
    config.pop('use_label_encoder', None)
    return XGBClassifier(**config)


def create_rf_model(config=None):
    """Create Random Forest classifier."""
    config = config or RF_CONFIG
    return RandomForestClassifier(**config)


def create_lr_model(config=None):
    """Create Logistic Regression classifier."""
    config = config or LR_CONFIG
    return LogisticRegression(**config)


def create_knn_model(config=None):
    """Create KNN classifier."""
    config = config or KNN_CONFIG
    return KNeighborsClassifier(**config)


def create_nb_model(config=None):
    """Create Naive Bayes classifier."""
    config = config or NB_CONFIG
    return GaussianNB(**config)


def create_et_model(config=None):
    """Create Extra Trees classifier."""
    config = config or ET_CONFIG
    return ExtraTreesClassifier(**config)


def create_voting_model(models=None):
    """Create Voting Classifier."""
    if models is None:
        # Default ensemble
        models = [
            ('ann', create_ann_model()),
            ('rf', create_rf_model()),
            ('svm', create_svm_model())
        ]
    return VotingClassifier(estimators=models, voting=ENSEMBLE_CONFIG['voting'])


def create_stacking_model(models=None, final_estimator=None):
    """Create Stacking Classifier."""
    if models is None:
        models = [
            ('ann', create_ann_model()),
            ('rf', create_rf_model()),
            ('svm', create_svm_model())
        ]
    if final_estimator is None:
        final_estimator = LogisticRegression()
    
    return StackingClassifier(estimators=models, final_estimator=final_estimator)


def get_model_by_name(name, input_dim=None):
    """
    Get model by name.
    
    Args:
        name: Model name ('ann', 'svm', 'xgboost', 'rf', 'lr', 'knn', 'nb', 'et', 'voting', 'stacking')
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
    elif name == 'lr':
        return create_lr_model()
    elif name == 'knn':
        return create_knn_model()
    elif name == 'nb':
        return create_nb_model()
    elif name == 'et':
        return create_et_model()
    elif name == 'voting':
        return create_voting_model()
    elif name == 'stacking':
        return create_stacking_model()
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
