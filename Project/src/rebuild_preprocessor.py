"""
Rebuild Preprocessing Pipeline using RAW training data
The original preprocessor was fitted on already-scaled data, which is incorrect.
This script creates the correct preprocessor by fitting on raw X_train data.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Goes up to workspace root
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'

def rebuild_preprocessor():
    """
    Rebuild the preprocessing pipeline using RAW (unscaled) training data
    """
    
    print("\n" + "="*80)
    print("REBUILDING PREPROCESSING PIPELINE WITH RAW DATA")
    print("="*80)
    
    # Load RAW training data (not processed/scaled)
    print("\n[STEP 1] Loading RAW training data...")
    X_train = pd.read_csv(DATA_DIR / 'X_train.csv')
    print(f"  ✓ Loaded X_train (raw): {X_train.shape}")
    
    # Get features that were used in training (from processed CSV for reference)
    print("\n[STEP 2] Loading reference features...")
    X_train_processed = pd.read_csv(DATA_DIR / 'X_train_processed.csv')
    feature_names = list(X_train_processed.columns)
    print(f"  ✓ Training features ({len(feature_names)}): {feature_names[:5]}...")
    
    # Filter raw data to only include training features
    print("\n[STEP 3] Filtering raw data to training features...")
    X_train_filtered = X_train[[col for col in feature_names if col in X_train.columns]].copy()
    
    # Convert ALL columns to numeric (handles string dates, strings, etc)
    print("\n[STEP 4] Converting all columns to numeric...")
    for col in X_train_filtered.columns:
        X_train_filtered[col] = pd.to_numeric(X_train_filtered[col], errors='coerce').fillna(0)
        if col in ['FileAlignment', 'Machine', 'NumberOfRvaAndSizes', 'SizeOfHeaders']:
            print(f"  ✓ Converted categorical {col}: unique values = {X_train_filtered[col].nunique()}")
    
    print(f"  ✓ Filtered data shape: {X_train_filtered.shape}")
    print(f"  ✓ Data dtypes: all numeric = {X_train_filtered.dtypes.unique()}")
    
    # Create StandardScaler on RAW data
    print("\n[STEP 5] Fitting StandardScaler on raw training data...")
    scaler = StandardScaler()
    scaler.fit(X_train_filtered)
    print(f"  ✓ Scaler fitted")
    print(f"    - Mean (first 5): {scaler.mean_[:5]}")
    print(f"    - Scale (first 5): {scaler.scale_[:5]}")
    
    # Identify categorical vs numeric
    categorical_features = ['FileAlignment', 'Machine', 'NumberOfRvaAndSizes', 'SizeOfHeaders']
    numeric_features = [col for col in feature_names if col not in categorical_features and col in X_train_filtered.columns]
    categorical_features_found = [col for col in categorical_features if col in X_train_filtered.columns]
    
    # Create preprocessor dictionary
    preprocessor_data = {
        'preprocessor': scaler,
        'feature_names': feature_names,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features_found,
        'n_features': len(feature_names)
    }
    
    # Save preprocessor
    print("\n[STEP 6] Saving preprocessor...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(MODELS_DIR / 'preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor_data, f)
    
    print(f"  ✓ Saved to: {MODELS_DIR / 'preprocessor.pkl'}")
    
    # Verify by loading
    print("\n[STEP 7] Verifying preprocessor...")
    with open(MODELS_DIR / 'preprocessor.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
    
    print(f"  ✓ Features: {len(loaded_data['feature_names'])}")
    print(f"  ✓ Numeric: {len(loaded_data['numeric_features'])}")
    print(f"  ✓ Categorical: {loaded_data['categorical_features']}")
    
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE REBUILT SUCCESSFULLY")
    print("="*80 + "\n")
    
    return preprocessor_data

if __name__ == '__main__':
    rebuild_preprocessor()
