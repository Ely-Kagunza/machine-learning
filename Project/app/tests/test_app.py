import pytest
import json
from pathlib import Path
import sys
from unittest.mock import Mock, patch

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

class TestBasic:
    """Basic functionality tests"""
    
    def test_health_check(self, client):
        """Test health endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.json
        assert 'status' in data
        assert data['status'] in ['ok', 'healthy']

class TestAPI:
    """API endpoint tests"""
    
    def test_features_endpoint(self, client):
        """Test features endpoint returns feature list"""
        response = client.get('/api/features')
        assert response.status_code == 200
        data = response.json
        assert ('features' in data or 'all_features' in data)
        features = data.get('features') or data.get('all_features')
        assert isinstance(features, list)
        assert len(features) > 0

    def test_model_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get('/api/model-info')
        assert response.status_code == 200
        data = response.json
        assert 'model_type' in data
        assert ('num_features' in data or 'features' in data)
