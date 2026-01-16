"""
Flask Web Application for Malware Detection
Week 3: Deploy LightGBM model with REST API and web interface
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random

# ============================================================================
# CONFIGURATION
# ============================================================================
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Goes up to workspace root
MODEL_PATH = PROJECT_ROOT / 'models' / 'best_model.pkl'
PREPROCESSOR_PATH = PROJECT_ROOT / 'models' / 'preprocessor.pkl'
DATA_PATH = PROJECT_ROOT / 'data'

# Model and preprocessor (loaded on startup)
MODEL = None
PREPROCESSOR = None
FEATURE_NAMES = None
CATEGORICAL_FEATURES = None
NUMERIC_FEATURES = None

# ============================================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================================
def load_model_and_preprocessor():
    """Load the trained model and preprocessing pipeline"""
    global MODEL, PREPROCESSOR, FEATURE_NAMES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
    
    try:
        # Load best model (LightGBM)
        with open(MODEL_PATH, 'rb') as f:
            MODEL = pickle.load(f)
        print(f"✓ Model loaded from {MODEL_PATH}")
        
        # Load preprocessor
        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        PREPROCESSOR = preprocessor_data['preprocessor']
        FEATURE_NAMES = preprocessor_data['feature_names']
        CATEGORICAL_FEATURES = preprocessor_data['categorical_features']
        NUMERIC_FEATURES = preprocessor_data['numeric_features']
        
        print(f"✓ Preprocessor loaded from {PREPROCESSOR_PATH}")
        print(f"  - Features: {len(FEATURE_NAMES)}")
        print(f"  - Categorical: {CATEGORICAL_FEATURES}")
        print(f"  - Numeric: {NUMERIC_FEATURES}")
        
    except Exception as e:
        print(f"✗ Error loading model/preprocessor: {e}")
        raise

# Load on app startup
load_model_and_preprocessor()

# ============================================================================
# SAMPLE DATA FUNCTIONS
# ============================================================================
def load_sample_data_from_file(sample_type='goodware'):
    """
    Load real sample data from training set (RAW, before preprocessing)
    
    Args:
        sample_type: 'goodware' (mean), 'malware' (mean), or 'random' (random goodware)
    
    Returns:
        Dictionary with feature values
    """
    try:
        # Load RAW data (before preprocessing) so it can be preprocessed correctly
        X_train = pd.read_csv(DATA_PATH / 'X_train.csv')
        y_train = pd.read_csv(DATA_PATH / 'y_train.csv').squeeze()
        
        # Select only the features that the model expects
        # First, filter to only columns that exist in FEATURE_NAMES
        available_features = [col for col in FEATURE_NAMES if col in X_train.columns]
        X_train_filtered = X_train[available_features].copy()
        
        # Convert categorical features to numeric (they might be stored as objects/strings)
        for cat_feature in CATEGORICAL_FEATURES:
            if cat_feature in X_train_filtered.columns:
                X_train_filtered[cat_feature] = pd.to_numeric(X_train_filtered[cat_feature], errors='coerce').fillna(0)
        
        # Separate numeric and categorical columns (after conversion)
        numeric_cols = X_train_filtered.select_dtypes(include=[np.number]).columns.tolist()
        X_train_numeric = X_train_filtered[numeric_cols]
        
        if sample_type == 'goodware':
            # Mean values of goodware (class 0)
            goodware_mask = y_train == 0
            goodware_samples = X_train_numeric[goodware_mask]
            # Use mean for numeric features
            sample = goodware_samples.mean().to_dict()
            # For categorical features, use mode (most common value) instead of mean
            for cat_feature in CATEGORICAL_FEATURES:
                if cat_feature in goodware_samples.columns:
                    mode_val = goodware_samples[cat_feature].mode()
                    sample[cat_feature] = int(mode_val.iloc[0]) if len(mode_val) > 0 else 0
            sample_name = "Mean Goodware Sample"
            
        elif sample_type == 'malware':
            # Use a REAL malware sample (random row) instead of class mean
            malware_indices = y_train[y_train == 1].index.tolist()
            random_idx = random.choice(malware_indices)
            sample = X_train_numeric.loc[random_idx].to_dict()
            # Ensure categorical features are ints
            for cat_feature in CATEGORICAL_FEATURES:
                if cat_feature in sample:
                    sample[cat_feature] = int(sample[cat_feature])
            sample_name = f"Random Malware Sample #{random_idx}"
            
        elif sample_type == 'random':
            # Random sample across both classes
            all_indices = y_train.index.tolist()
            random_idx = random.choice(all_indices)
            sample = X_train_numeric.loc[random_idx].to_dict()
            # For categorical features, convert to int
            for cat_feature in CATEGORICAL_FEATURES:
                if cat_feature in sample:
                    sample[cat_feature] = int(sample[cat_feature])
            label = int(y_train.loc[random_idx])
            label_name = 'Goodware' if label == 0 else 'Malware'
            sample_name = f"Random {label_name} Sample #{random_idx}"
        
        # Ensure all required features are present (fill missing with 0)
        for feature in FEATURE_NAMES:
            if feature not in sample:
                sample[feature] = 0.0
        
        # Debug: Print what we got
        print(f"\n[DEBUG] Sample type: {sample_type}")
        print(f"[DEBUG] Features in sample before conversion: {list(sample.keys())}")
        print(f"[DEBUG] Categorical features: {CATEGORICAL_FEATURES}")
        
        # Convert all values to proper numeric format for JSON serialization
        result_sample = {}
        for feature in FEATURE_NAMES:
            value = sample.get(feature, 0.0)
            try:
                # All features should be converted to proper numeric format
                if feature in CATEGORICAL_FEATURES:
                    # Categorical features must be integers
                    result_sample[feature] = int(round(float(value)))
                else:
                    # Numeric features as floats
                    result_sample[feature] = float(value)
            except (ValueError, TypeError):
                result_sample[feature] = 0
        
        print(f"[DEBUG] Categorical values (final): {[(k, result_sample[k]) for k in CATEGORICAL_FEATURES if k in result_sample]}\n")
        
        return {
            'sample': result_sample,
            'sample_name': sample_name,
            'features': len(FEATURE_NAMES)
        }
    
    except Exception as e:
        return {'error': f'Error loading sample data: {str(e)}'}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def preprocess_input(data_dict):
    """
    Preprocess input data to match training format
    
    Args:
        data_dict: Dictionary with feature names as keys
    
    Returns:
        Preprocessed numpy array (1, n_features)
    """
    try:
        # Create DataFrame with proper feature order
        df = pd.DataFrame([data_dict], columns=FEATURE_NAMES)
        
        print(f"[DEBUG] DataFrame shape: {df.shape}")
        print(f"[DEBUG] DataFrame columns: {list(df.columns)}")
        print(f"[DEBUG] First few values: {df.iloc[0, :5].to_dict()}")
        
        # Apply preprocessing
        X_processed = PREPROCESSOR.transform(df)
        
        print(f"[DEBUG] Processed shape: {X_processed.shape}")
        
        return X_processed
    
    except Exception as e:
        raise ValueError(f"Preprocessing error: {e}")

def make_prediction(X_processed):
    """
    Make prediction with the model
    
    Args:
        X_processed: Preprocessed feature array (1, n_features)
    
    Returns:
        Dictionary with prediction and probability
    """
    try:
        # Wrap scaled features with column names to match training
        X_df = pd.DataFrame(X_processed, columns=FEATURE_NAMES)

        # Get prediction
        y_pred = MODEL.predict(X_df)[0]
        
        # Get prediction probability
        y_proba = MODEL.predict_proba(X_df)[0]
        
        return {
            'prediction': int(y_pred),
            'probability_goodware': float(y_proba[0]),
            'probability_malware': float(y_proba[1]),
            'confidence': float(max(y_proba)) * 100
        }
    
    except Exception as e:
        raise ValueError(f"Prediction error: {e}")

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def home():
    """Render home page with input form"""
    return render_template('index.html', 
                         features=FEATURE_NAMES,
                         numeric_features=NUMERIC_FEATURES,
                         categorical_features=CATEGORICAL_FEATURES)

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for predictions
    Expects JSON with feature values
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required features
        missing_features = [f for f in FEATURE_NAMES if f not in data]
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'}), 400
        
        # Extract only required features in correct order
        input_data = {f: data[f] for f in FEATURE_NAMES}
        
        # Preprocess
        X_processed = preprocess_input(input_data)
        
        # Make prediction
        result = make_prediction(X_processed)
        
        # Interpret result
        if result['prediction'] == 0:
            result['classification'] = 'GOODWARE ✓'
            result['status'] = 'Safe'
            result['color'] = 'success'
        else:
            result['classification'] = 'MALWARE ⚠️'
            result['status'] = 'Dangerous'
            result['color'] = 'danger'
        
        return jsonify(result), 200
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get feature information"""
    return jsonify({
        'total_features': len(FEATURE_NAMES),
        'numeric_features': NUMERIC_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'all_features': FEATURE_NAMES
    }), 200

@app.route('/api/sample/<sample_type>', methods=['GET'])
def get_sample(sample_type):
    """
    Get sample data for testing
    sample_type: goodware, malware, or random
    """
    if sample_type not in ['goodware', 'malware', 'random']:
        return jsonify({'error': 'Invalid sample type. Use: goodware, malware, or random'}), 400
    
    result = load_sample_data_from_file(sample_type)
    
    if 'error' in result:
        return jsonify(result), 500
    
    return jsonify(result), 200

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': 'LightGBM',
        'cv_auc': 0.9957,
        'test_auc': 0.8678,
        'test_accuracy': 0.9965,
        'features': len(FEATURE_NAMES),
        'status': 'Production Ready'
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'preprocessor_loaded': PREPROCESSOR is not None
    }), 200

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("MALWARE DETECTION FLASK WEB APP STARTING")
    print("="*80)
    print(f"Model: LightGBM")
    print(f"CV AUC: 0.9957")
    print(f"Status: Ready for predictions")
    print("="*80)
    print("\nStarting Flask app on http://127.0.0.1:5000")
    print("Press CTRL+C to stop the server\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
