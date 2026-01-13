# Lung Cancer Prediction

A machine learning project comparing four classification models for lung cancer prediction using 5-fold cross-validation.

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/MacOS
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training (Steps to run)
```bash
# Train models on original data (Fold)
python main.py train

# Train with synthetic missing values (Synthetic)
python main.py train --missing

# Run complete pipeline (Fold & Synthetic)
python main.py train --all
```

### Inference (Making Predictions)
```bash
# Predict using the saved best model
python main.py predict

# Predict on a custom CSV file
python main.py predict --input my_data.csv --output result.csv
```

## File Descriptions

| File | Description |
|------|-------------|
| `main.py` | Unified entry point for training and inference |
| `scripts/train.py` | Main training logic with cross-validation |
| `scripts/inference.py` | Prediction logic using saved models |
| `scripts/models.py` | Model architectures (ANN, SVM, XGBoost, Random Forest) |
| `scripts/data_loader.py` | Data loading, encoding, and scaling utilities |
| `scripts/missing_values.py` | Handling and imputing missing data |
| `scripts/utils.py` | Calculation metrics and plotting functions |
| `scripts/config.py` | Hyperparameters and path configurations |

## Hardware Requirements

- **Minimum**: CPU with 4GB RAM
- **Recommended**: GPU support is optional but can speed up ANN training.
- **Project Structure**: Very modular and easy to understand, designed to run efficiently on standard CPUs.
- **Tested on**: Windows 10/11, Python 3.8+
