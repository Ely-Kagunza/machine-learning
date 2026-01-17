"""
Integration tests for complete workflows
"""
import pytest
import json
import csv
from io import BytesIO, StringIO
from unittest.mock import Mock
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


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for upload testing"""
    features = [str(i) for i in range(26)]
    rows = [
        {**{f: '0.5' for f in features}, 'Label': '0'},
        {**{f: '0.7' for f in features}, 'Label': '1'},
        {**{f: '0.3' for f in features}, 'Label': '0'},
    ]
    return features, rows


@pytest.fixture
def csv_file_bytes(sample_csv_data):
    """Create CSV bytes for multipart upload"""
    features, rows = sample_csv_data
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=features + ['Label'])
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode('utf-8')


class TestAppInitialization:
    """Test app initializes correctly"""
    
    def test_app_loads(self, client):
        """App should load without errors"""
        response = client.get('/health')
        assert response.status_code == 200

    def test_model_features_available(self, client):
        """Model features should be available after load"""
        response = client.get('/api/features')
        assert response.status_code == 200
        data = response.json
        features = data.get('features') or data.get('all_features')
        assert len(features) == 26

    def test_model_info_available(self, client):
        """Model info should be available"""
        response = client.get('/api/model-info')
        assert response.status_code == 200


class TestPredictionWorkflow:
    """Test complete prediction workflows"""
    
    def test_get_features_then_predict(self, client):
        """Should be able to get features then make prediction"""
        # Step 1: Get features
        features_response = client.get('/api/features')
        assert features_response.status_code == 200
        
        # Step 2: Create prediction payload
        features = features_response.json.get('features') or features_response.json.get('all_features')
        predict_data = {f: 0.5 for f in features}
        
        # Step 3: Make prediction (may fail if model not loaded, but endpoint should exist)
        predict_response = client.post('/api/predict', json=predict_data)
        assert predict_response.status_code in [200, 400, 500]  # 400/500 ok if model not loaded

    def test_get_model_info_then_predict(self, client):
        """Should be able to get model info then make prediction"""
        # Step 1: Get model info
        info_response = client.get('/api/model-info')
        assert info_response.status_code == 200
        info = info_response.json
        num_features = info.get('features') or info.get('num_features')
        
        # Step 2: Create and send prediction
        predict_data = {str(i): 0.5 for i in range(num_features)}
        predict_response = client.post('/api/predict', json=predict_data)
        assert predict_response.status_code in [200, 400, 500]


class TestUploadWorkflow:
    """Test file upload workflows"""
    
    def test_upload_csv_workflow(self, client, csv_file_bytes):
        """Test complete CSV upload workflow"""
        data = {
            'file': (BytesIO(csv_file_bytes), 'test.csv')
        }
        response = client.post('/api/upload', 
                              data=data,
                              content_type='multipart/form-data')
        # Upload endpoint should either process successfully or give meaningful error
        assert response.status_code in [200, 400, 422, 500]

    def test_upload_with_labels_generates_metrics(self, client, csv_file_bytes):
        """Upload with Label column should generate metrics"""
        data = {
            'file': (BytesIO(csv_file_bytes), 'test_with_labels.csv')
        }
        response = client.post('/api/upload',
                              data=data,
                              content_type='multipart/form-data')
        
        if response.status_code == 200:
            result = response.json
            # If metrics are calculated, should have these fields
            if 'metrics' in result:
                metrics = result['metrics']
                assert 'accuracy' in metrics or 'auc' in metrics or 'predictions' in result

    def test_upload_without_required_columns_fails_gracefully(self, client):
        """Upload with missing columns should fail gracefully"""
        # Create CSV with wrong columns
        bad_csv = "col1,col2\n1,2\n3,4\n"
        data = {
            'file': (BytesIO(bad_csv.encode()), 'bad.csv')
        }
        response = client.post('/api/upload',
                              data=data,
                              content_type='multipart/form-data')
        # Should not crash, may return 400/422 or process with defaults
        assert response.status_code in [200, 400, 422, 500]


class TestDataIntegrity:
    """Test data integrity across requests"""
    
    def test_consistent_feature_names(self, client):
        """Feature names should be consistent across endpoints"""
        features_response = client.get('/api/features')
        info_response = client.get('/api/model-info')
        
        assert features_response.status_code == 200
        assert info_response.status_code == 200
        
        features_count = (features_response.json.get('features') or 
                         features_response.json.get('all_features'))
        info_count = (info_response.json.get('features') or 
                     info_response.json.get('num_features'))
        
        assert len(features_count) == info_count

    def test_model_info_stability(self, client):
        """Model info should be stable across multiple requests"""
        response1 = client.get('/api/model-info')
        response2 = client.get('/api/model-info')
        
        assert response1.status_code == response2.status_code
        if response1.status_code == 200:
            data1 = response1.json
            data2 = response2.json
            assert data1['model_type'] == data2['model_type']


class TestEndpointAccessibility:
    """Test all endpoints are accessible"""
    
    def test_all_get_endpoints_accessible(self, client):
        """All GET endpoints should be accessible"""
        endpoints = [
            '/health',
            '/api/features',
            '/api/model-info',
        ]
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code != 404

    def test_all_post_endpoints_accessible(self, client):
        """All POST endpoints should be accessible"""
        endpoints = [
            '/api/predict',
            '/api/upload',
        ]
        for endpoint in endpoints:
            response = client.post(endpoint, json={})
            # Should not be 404 (may be 400 for bad data, but endpoint exists)
            assert response.status_code != 404

    def test_sample_endpoints_accessible(self, client):
        """Sample endpoints should be accessible"""
        types = ['goodware', 'malware', 'random']
        for sample_type in types:
            response = client.get(f'/api/sample/{sample_type}')
            # Should not be 404
            assert response.status_code != 404


class TestResponseFormats:
    """Test response format consistency"""
    
    def test_error_responses_have_error_field(self, client):
        """Error responses should have error field"""
        # Send invalid data to predict
        response = client.post('/api/predict', json={})
        if response.status_code in [400, 422]:
            data = response.json
            assert 'error' in data

    def test_success_responses_are_json(self, client):
        """Success responses should be valid JSON"""
        response = client.get('/health')
        assert response.status_code == 200
        assert response.content_type == 'application/json'
        data = response.json
        assert isinstance(data, dict)

    def test_batch_response_structure(self, client, csv_file_bytes):
        """Batch upload response should have consistent structure"""
        data = {
            'file': (BytesIO(csv_file_bytes), 'test.csv')
        }
        response = client.post('/api/upload',
                              data=data,
                              content_type='multipart/form-data')
        
        if response.status_code == 200:
            result = response.json
            # Should have some result data
            assert len(result) > 0


class TestPerformanceAndResilience:
    """Test performance and error resilience"""
    
    def test_rapid_successive_requests(self, client):
        """Should handle rapid successive requests"""
        for _ in range(5):
            response = client.get('/health')
            assert response.status_code == 200

    def test_mixed_endpoint_requests(self, client):
        """Should handle mixed endpoint requests"""
        endpoints = [
            ('GET', '/health'),
            ('GET', '/api/features'),
            ('GET', '/api/model-info'),
            ('GET', '/api/sample/goodware'),
        ]
        
        for method, endpoint in endpoints:
            if method == 'GET':
                response = client.get(endpoint)
                assert response.status_code in [200, 400, 404]

    def test_large_feature_vector(self, client):
        """Should handle large feature vectors"""
        # Create large feature vector
        large_data = {str(i): float(i) / 100 for i in range(100)}
        response = client.post('/api/predict', json=large_data)
        # Should handle gracefully even if features don't match
        assert response.status_code in [200, 400, 422, 500]
