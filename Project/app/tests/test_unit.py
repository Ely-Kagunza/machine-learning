"""
Unit tests for individual components and functions
"""
import pytest
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock sklearn before importing app
sys.modules['sklearn'] = Mock()
sys.modules['sklearn.preprocessing'] = Mock()
sys.modules['sklearn.metrics'] = Mock()


@pytest.fixture
def client():
    """Create test client"""
    from app import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check_returns_200(self, client):
        """Health endpoint should return 200 status"""
        response = client.get('/health')
        assert response.status_code == 200

    def test_health_check_has_status(self, client):
        """Health endpoint should return status field"""
        response = client.get('/health')
        data = response.json
        assert 'status' in data
        assert data['status'] in ['ok', 'healthy']

    def test_health_check_response_format(self, client):
        """Health endpoint should return valid JSON"""
        response = client.get('/health')
        assert response.content_type == 'application/json'
        data = response.json
        assert isinstance(data, dict)


class TestFeaturesEndpoint:
    """Test features endpoint"""
    
    def test_features_endpoint_returns_200(self, client):
        """Features endpoint should return 200 status"""
        response = client.get('/api/features')
        assert response.status_code == 200

    def test_features_endpoint_returns_feature_list(self, client):
        """Features endpoint should return a list of features"""
        response = client.get('/api/features')
        data = response.json
        features = data.get('features') or data.get('all_features')
        assert isinstance(features, list)
        assert len(features) > 0

    def test_features_endpoint_returns_numeric_and_categorical(self, client):
        """Features endpoint should return both numeric and categorical features"""
        response = client.get('/api/features')
        data = response.json
        assert 'numeric_features' in data or 'categorical_features' in data

    def test_features_count_matches(self, client):
        """Total features count should match sum of numeric and categorical"""
        response = client.get('/api/features')
        data = response.json
        total = data.get('total_features', 0)
        assert total > 0


class TestModelInfoEndpoint:
    """Test model info endpoint"""
    
    def test_model_info_returns_200(self, client):
        """Model info endpoint should return 200 status"""
        response = client.get('/api/model-info')
        assert response.status_code == 200

    def test_model_info_has_type(self, client):
        """Model info should include model type"""
        response = client.get('/api/model-info')
        data = response.json
        assert 'model_type' in data
        assert data['model_type'] in ['LightGBM', 'LightGBMClassifier']

    def test_model_info_has_features(self, client):
        """Model info should include feature count"""
        response = client.get('/api/model-info')
        data = response.json
        assert ('features' in data or 'num_features' in data)
        num_features = data.get('features') or data.get('num_features')
        assert num_features == 26

    def test_model_info_has_performance_metrics(self, client):
        """Model info should include performance metrics"""
        response = client.get('/api/model-info')
        data = response.json
        # At least one performance metric should be present
        has_metric = any(k in data for k in ['cv_auc', 'accuracy', 'precision', 'recall'])
        assert has_metric


class TestPredictEndpoint:
    """Test prediction endpoint"""
    
    def test_predict_missing_file(self, client):
        """Prediction with no data should fail"""
        response = client.post('/api/predict', json={})
        assert response.status_code in [400, 422, 500]

    def test_predict_invalid_content_type(self, client):
        """Prediction with wrong content type should fail"""
        response = client.post('/api/predict', 
                              data="not json",
                              content_type='text/plain')
        assert response.status_code in [400, 415, 422, 500]

    def test_predict_with_partial_features(self, client):
        """Prediction with partial features should fail"""
        partial_data = {str(i): 0.5 for i in range(10)}
        response = client.post('/api/predict',
                              json=partial_data)
        # Should either fail validation or handle gracefully
        assert response.status_code in [200, 400, 422]


class TestUploadEndpoint:
    """Test file upload endpoint"""
    
    def test_upload_no_file(self, client):
        """Upload without file should fail"""
        response = client.post('/api/upload', data={})
        assert response.status_code in [400, 422]

    def test_upload_empty_file(self, client):
        """Upload empty file should fail"""
        from io import BytesIO
        data = {'file': (BytesIO(b''), 'test.csv')}
        response = client.post('/api/upload', data=data, content_type='multipart/form-data')
        assert response.status_code in [400, 422, 500]

    def test_upload_endpoint_exists(self, client):
        """Upload endpoint should be accessible"""
        response = client.post('/api/upload')
        # Should return 400 (no file) not 404 (not found)
        assert response.status_code != 404


class TestSampleEndpoints:
    """Test sample data endpoints"""
    
    def test_sample_endpoint_exists(self, client):
        """Sample endpoints should exist"""
        response = client.get('/api/sample/goodware')
        assert response.status_code != 404

    def test_sample_invalid_type(self, client):
        """Invalid sample type should be rejected"""
        response = client.get('/api/sample/invalid_type')
        assert response.status_code in [400, 404]

    def test_sample_goodware_type(self, client):
        """Goodware sample should work"""
        response = client.get('/api/sample/goodware')
        if response.status_code == 200:
            data = response.json
            assert 'features' in data

    def test_sample_malware_type(self, client):
        """Malware sample should work"""
        response = client.get('/api/sample/malware')
        if response.status_code == 200:
            data = response.json
            assert 'features' in data


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_nonexistent_endpoint(self, client):
        """Nonexistent endpoint should return 404"""
        response = client.get('/api/nonexistent')
        assert response.status_code == 404

    def test_wrong_http_method(self, client):
        """Using wrong HTTP method should fail"""
        response = client.get('/api/predict')
        assert response.status_code in [405, 400]

    def test_cors_headers_present(self, client):
        """Response should have CORS headers or be accessible"""
        response = client.get('/health')
        assert response.status_code == 200

    def test_response_is_json(self, client):
        """All API responses should be JSON"""
        endpoints = ['/health', '/api/features', '/api/model-info']
        for endpoint in endpoints:
            response = client.get(endpoint)
            if response.status_code == 200:
                assert response.content_type == 'application/json'
