"""
Preprocessing Pipeline for Malware Detection
Recreates the StandardScaler pipeline used for training
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

def identify_feature_types():
    """Identify numeric and categorical features from training data"""
    X_train = pd.read_csv(DATA_DIR / 'X_train_processed.csv')
    
    numeric_features = []
    categorical_features = []
    
    for col in X_train.columns:
        # Check unique values - if 2-20 unique values, likely categorical
        unique_count = X_train[col].nunique()
        
        if unique_count <= 10 or X_train[col].dtype == 'object':
            categorical_features.append(col)
        else:
            numeric_features.append(col)
    
    return numeric_features, categorical_features

def create_preprocessor():
    """
    Create and save the preprocessing pipeline
    The training data was already preprocessed (scaled), but we need the pipeline
    for new predictions
    """
    
    print("\n" + "="*80)
    print("CREATING PREPROCESSING PIPELINE")
    print("="*80)
    
    # Load training data
    print("\n[STEP 1] Loading training data...")
    X_train = pd.read_csv(DATA_DIR / 'X_train_processed.csv')
    print(f"  ✓ Loaded X_train: {X_train.shape}")
    
    # Identify features
    print("\n[STEP 2] Identifying feature types...")
    numeric_features, categorical_features = identify_feature_types()
    print(f"  ✓ Numeric features ({len(numeric_features)}): {numeric_features[:5]}...")
    print(f"  ✓ Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Create StandardScaler
    print("\n[STEP 3] Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    print(f"  ✓ Scaler fitted")
    print(f"    - Mean shape: {scaler.mean_.shape}")
    print(f"    - Scale shape: {scaler.scale_.shape}")
    
    # Create preprocessor dictionary
    preprocessor_data = {
        'preprocessor': scaler,
        'feature_names': list(X_train.columns),
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'n_features': X_train.shape[1]
    }
    
    # Save preprocessor
    print("\n[STEP 4] Saving preprocessor...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(MODELS_DIR / 'preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor_data, f)
    
    print(f"  ✓ Saved to: {MODELS_DIR / 'preprocessor.pkl'}")
    
    # Verify by loading
    print("\n[STEP 5] Verifying preprocessor...")
    with open(MODELS_DIR / 'preprocessor.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
    
    print(f"  ✓ Features: {len(loaded_data['feature_names'])}")
    print(f"  ✓ Numeric: {len(loaded_data['numeric_features'])}")
    print(f"  ✓ Categorical: {len(loaded_data['categorical_features'])}")
    
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE CREATED SUCCESSFULLY")
    print("="*80 + "\n")
    
    return preprocessor_data

if __name__ == '__main__':
    preprocessor_data = create_preprocessor()
    
    # Print summary
    print("Preprocessor Information:")
    print(f"  - Features: {preprocessor_data['feature_names']}")
    print(f"  - Total Features: {preprocessor_data['n_features']}")
