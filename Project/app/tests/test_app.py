import pytest
import json
from pathlib import Path
import sys

# Add parent directory to path so we can import app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app, MODEL, PREPROCESSOR, FEATURE_NAMES

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

class TestBasic:
    """Basic functionality tests"""
    
    def test_health_check(self, client):
        """Test health endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.json
        assert data['status'] == 'ok'

    def test_model_loaded(self):
        """Test model and preprocessor are loaded"""
        assert MODEL is not None, "Model not loaded"
        assert PREPROCESSOR is not None, "Preprocessor not loaded"
        assert FEATURE_NAMES is not None, "Feature names not loaded"

class TestAPI:
    """API endpoint tests"""
    
    def test_features_endpoint(self, client):
        """Test features endpoint returns feature list"""
        response = client.get('/api/features')
        assert response.status_code == 200
        data = response.json
        assert 'features' in data
        assert isinstance(data['features'], list)
        assert len(data['features']) > 0

    def test_model_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get('/api/model-info')
        assert response.status_code == 200
        data = response.json
        assert 'model_type' in data
        assert 'num_features' in data

    def test_predict_valid_input(self, client):
        """Test prediction endpoint with valid data"""
        # Create dummy feature values matching model's 26 features
        sample_data = {str(i): 0.5 for i in range(26)}
        response = client.post('/api/predict', 
                               json=sample_data,
                               content_type='application/json')
        assert response.status_code == 200
        data = response.json
        assert 'prediction' in data
        assert 'probability' in data
        assert data['prediction'] in [0, 1]
        assert 0 <= data['probability'] <= 1

    def test_predict_missing_features(self, client):
        """Test prediction with missing features"""
        incomplete_data = {str(i): 0.5 for i in range(10)}
        response = client.post('/api/predict', 
                               json=incomplete_data,
                               content_type='application/json')
        assert response.status_code in [400, 422]
        data = response.json
        assert 'error' in data

class TestUpload:
    """File upload and batch processing tests"""
    
    def test_upload_endpoint_exists(self, client):
        """Test upload endpoint is available"""
        response = client.post('/api/upload', data={})
        # Should fail with 400 (no file) rather than 404
        assert response.status_code in [400, 422]

class TestSamples:
    """Sample data endpoint tests"""
    
    def test_sample_goodware(self, client):
        """Test goodware sample endpoint"""
        response = client.get('/api/sample/goodware')
        assert response.status_code == 200
        data = response.json
        assert 'features' in data

    def test_sample_malware(self, client):
        """Test malware sample endpoint"""
        response = client.get('/api/sample/malware')
        assert response.status_code == 200
        data = response.json
        assert 'features' in data

    def test_sample_invalid_type(self, client):
        """Test invalid sample type"""
        response = client.get('/api/sample/invalid')
        assert response.status_code == 400
        data = response.json
        assert 'error' in data
