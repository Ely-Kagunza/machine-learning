"""
Preprocessing Pipeline for Brazilian Malware Detection
Week 1: Data Preprocessing and Feature Engineering

This module implements a robust preprocessing pipeline that:
1. Removes high-missing columns
2. Removes constant variance features
3. Handles missing values
4. Encodes categorical features
5. Scales numeric features
6. Fits ONLY on training data (prevents data leakage)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')


class MalwarePreprocessor:
    """
    Preprocessing pipeline for Brazilian malware dataset.
    Ensures all preprocessing is fit on training data only.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.encoders = {}  # Store encoders for categorical columns
        self.numeric_features = None
        self.categorical_features = None
        self.features_to_drop = None
        self.is_fitted = False
        
    def fit(self, X_train):
        """
        Fit the preprocessing pipeline on training data ONLY.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features (not including target)
        
        Returns:
        --------
        self : MalwarePreprocessor
        """
        print("\n[PREPROCESSING] Fitting pipeline on training data...")
        
        # STEP 1: Identify features to drop
        self._identify_features_to_drop(X_train)
        X_train_clean = X_train.drop(columns=self.features_to_drop)
        print(f"  - Removed {len(self.features_to_drop)} problematic features")
        
        # STEP 2: Separate numeric and categorical features
        self.numeric_features = X_train_clean.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X_train_clean.select_dtypes(include=['object']).columns.tolist()
        
        print(f"  - Numeric features: {len(self.numeric_features)}")
        print(f"  - Categorical features: {len(self.categorical_features)}")
        
        # STEP 3: Fit scaler on numeric features
        if self.numeric_features:
            self.scaler.fit(X_train_clean[self.numeric_features])
            print(f"  - StandardScaler fit on {len(self.numeric_features)} numeric features")
        
        # STEP 4: Fit label encoders for categorical features
        for cat_col in self.categorical_features:
            encoder = LabelEncoder()
            # Fill NaN with 'MISSING' for encoding
            train_values = X_train_clean[cat_col].fillna('MISSING').astype(str)
            encoder.fit(train_values)
            self.encoders[cat_col] = encoder
        
        if self.encoders:
            print(f"  - LabelEncoders fit for {len(self.encoders)} categorical features")
        
        self.is_fitted = True
        print("  - Pipeline fitted successfully!")
        return self
    
    def transform(self, X):
        """
        Transform data using the fitted pipeline.
        Can be applied to both training and test data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features to transform
        
        Returns:
        --------
        X_transformed : pd.DataFrame
            Preprocessed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")
        
        X_copy = X.copy()
        
        # Remove problematic features
        X_copy = X_copy.drop(columns=self.features_to_drop)
        
        # Handle numeric features: scale
        if self.numeric_features:
            X_copy[self.numeric_features] = self.scaler.transform(X_copy[self.numeric_features])
        
        # Handle categorical features: encode
        for cat_col in self.categorical_features:
            # Fill NaN with 'MISSING' for consistency with training
            X_copy[cat_col] = X_copy[cat_col].fillna('MISSING').astype(str)
            
            # Encode labels - use LabelEncoder's transform which handles values it's seen
            encoder = self.encoders[cat_col]
            # Map values to encoder indices, handling unseen values
            def encode_value(x):
                try:
                    return encoder.transform([x])[0]
                except (ValueError, KeyError):
                    # For unseen values, map to the 'MISSING' encoding index
                    # This handles cases where test set has values training didn't see
                    if 'MISSING' in encoder.classes_:
                        return encoder.transform(['MISSING'])[0]
                    else:
                        # Fallback: return first class index (rare case)
                        return 0
            
            X_copy[cat_col] = X_copy[cat_col].apply(encode_value)
        
        return X_copy
    
    def fit_transform(self, X_train):
        """
        Fit and transform in one step (use only on training data!).
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        
        Returns:
        --------
        X_transformed : pd.DataFrame
            Preprocessed training features
        """
        self.fit(X_train)
        return self.transform(X_train)
    
    def _identify_features_to_drop(self, X):
        """
        Identify features that should be dropped:
        1. Columns with >95% missing values
        2. Columns with zero variance (constant values)
        """
        features_to_drop = []
        
        # Check for high-missing columns (>95%)
        missing_pct = (X.isnull().sum() / len(X) * 100)
        high_missing = missing_pct[missing_pct > 95].index.tolist()
        features_to_drop.extend(high_missing)
        
        # Check for constant/zero-variance features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].nunique() <= 1:  # 0 or 1 unique value
                features_to_drop.append(col)
        
        self.features_to_drop = list(set(features_to_drop))
    
    def save(self, filepath):
        """Save fitted preprocessor to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"  - Preprocessor saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load fitted preprocessor from disk."""
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"  - Preprocessor loaded from: {filepath}")
        return preprocessor


def main():
    """
    Main execution: fit preprocessor on training data and apply to both splits.
    """
    print("\n" + "="*80)
    print("MALWARE DETECTION - PREPROCESSING PIPELINE")
    print("="*80)
    
    # Load data (data directory is in root, not in src)
    base_path = Path(__file__).parent.parent  # Go up to project root
    data_dir = base_path / "data"
    
    print("\n[STEP 1] Loading train/test splits...")
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()
    
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - X_test shape: {X_test.shape}")
    print(f"  - y_train shape: {y_train.shape}")
    print(f"  - y_test shape: {y_test.shape}")
    
    # Initialize and fit preprocessor
    print("\n[STEP 2] Creating and fitting preprocessor (training data only)...")
    preprocessor = MalwarePreprocessor(random_state=42)
    preprocessor.fit(X_train)
    
    # Transform both splits
    print("\n[STEP 3] Transforming training and test data...")
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"  - X_train_processed shape: {X_train_processed.shape}")
    print(f"  - X_test_processed shape: {X_test_processed.shape}")
    
    # Verify no NaN values remain
    print("\n[STEP 4] Data quality checks...")
    print(f"  - NaN in X_train: {X_train_processed.isnull().sum().sum()}")
    print(f"  - NaN in X_test: {X_test_processed.isnull().sum().sum()}")
    print(f"  - Data types correct: {all(pd.api.types.is_numeric_dtype(X_train_processed[col]) for col in X_train_processed.columns)}")
    
    # Save processed data
    print("\n[STEP 5] Saving processed data...")
    X_train_processed.to_csv(data_dir / "X_train_processed.csv", index=False)
    X_test_processed.to_csv(data_dir / "X_test_processed.csv", index=False)
    y_train.to_csv(data_dir / "y_train_processed.csv", index=False)
    y_test.to_csv(data_dir / "y_test_processed.csv", index=False)
    
    # Save preprocessor for later use in Flask app
    print("\n[STEP 6] Saving preprocessor for production use...")
    preprocessor.save(data_dir / "preprocessor.pkl")
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print(f"Files saved to: {data_dir}")
    print("Ready for model training!")
    print("="*80 + "\n")
    
    return preprocessor, X_train_processed, X_test_processed, y_train, y_test


if __name__ == "__main__":
    preprocessor, X_train, X_test, y_train, y_test = main()
