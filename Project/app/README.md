# Flask Malware Detection Web App - Week 3

## Overview
This is a production-ready Flask web application for the LightGBM malware detection model trained in Week 2.

## Features
- **REST API** for predictions (`/api/predict`)
- **Web Interface** for interactive analysis
- **Model Information** endpoint (`/api/model-info`)
- **Feature Details** endpoint (`/api/features`)
- **Health Check** endpoint (`/health`)
- **Real-time Predictions** with confidence scores
- **Bootstrap UI** with responsive design

## Project Structure
```
app/
├── app.py                 # Flask application (main entry point)
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Web interface template
└── static/
    ├── style.css         # Stylesheet
    └── app.js            # JavaScript application logic
```

## Installation & Setup

### 1. Install Dependencies
```bash
cd Project/app
pip install -r requirements.txt
```

### 2. Run the Flask App
```bash
python app.py
```

The app will start on `http://127.0.0.1:5000`

## API Endpoints

### 1. Web Interface
- **GET** `/` - Main web interface with prediction form

### 2. Make Prediction
- **POST** `/api/predict`
- **Content-Type**: `application/json`
- **Body**: JSON with all 26 feature values

**Example Request:**
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "API_Count": 145,
    "String_Count": 89,
    ...
  }'
```

**Example Response:**
```json
{
  "prediction": 0,
  "probability_goodware": 0.98,
  "probability_malware": 0.02,
  "confidence": 98.0,
  "classification": "GOODWARE ✓",
  "status": "Safe",
  "color": "success"
}
```

### 3. Get Model Information
- **GET** `/api/model-info`

**Response:**
```json
{
  "model_type": "LightGBM",
  "cv_auc": 0.9957,
  "test_auc": 0.8678,
  "test_accuracy": 0.9965,
  "features": 26,
  "status": "Production Ready"
}
```

### 4. Get Feature Information
- **GET** `/api/features`

**Response:**
```json
{
  "total_features": 26,
  "numeric_features": [...],
  "categorical_features": [...],
  "all_features": [...]
}
```

### 5. Health Check
- **GET** `/health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true
}
```

## Model Details

| Metric | Value |
|--------|-------|
| **Algorithm** | LightGBM (Light Gradient Boosting Machine) |
| **CV AUC** | 0.9957 |
| **Test AUC** | 0.8678 |
| **Test Accuracy** | 99.65% |
| **Features** | 26 |
| **Training Samples** | 16,971 |
| **Test Samples** | 4,243 |
| **Class Balance** | 42% goodware, 58% malware |

## Usage Examples

### Python Client
```python
import requests
import json

url = "http://127.0.0.1:5000/api/predict"

data = {
    "API_Count": 145,
    "String_Count": 89,
    "Entropy": 6.2,
    ...
}

response = requests.post(url, json=data)
result = response.json()

print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']:.2f}%")
print(f"Probability Malware: {result['probability_malware']*100:.2f}%")
```

### JavaScript Client
```javascript
fetch('/api/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        API_Count: 145,
        String_Count: 89,
        ...
    })
})
.then(r => r.json())
.then(result => console.log(result))
```

### cURL
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d @sample_data.json
```

## Features (Input Variables)

### Numeric Features (14)
- API_Count
- String_Count
- Entropy
- Virtual_Size
- Raw_Size
- Timestamp
- Machine_Type
- Subsystem
- Characteristics
- Header_Size
- Code_Section_Entropy
- Data_Section_Entropy
- Imported_DLLs
- Exported_Functions

### Categorical Features (12)
- Legitimate_API
- Suspicious_String
- Packed
- Debug_Info
- Import_Table
- Export_Table
- Resource_Section
- Text_Section
- Data_Section
- Section_Count (encoded)

## Configuration

### Debug Mode
- Currently set to `debug=True` in `app.py`
- Change to `debug=False` for production

### Host and Port
- Host: `127.0.0.1` (localhost only)
- Port: `5000`

### For Production Deployment
```python
# Use production server
# gunicorn app:app --workers 4 --bind 0.0.0.0:5000
```

## Troubleshooting

### Model Not Loading
- Verify `models/best_model.pkl` exists
- Verify `models/preprocessor.pkl` exists
- Check file paths in `app.py`

### Prediction Errors
- Ensure all 26 features are provided in request
- Check feature data types (numeric vs categorical)
- Verify feature names match exactly

### Port Already in Use
```bash
# Change port in app.py or use:
python -m flask --app app run --port 5001
```

## Next Steps (Week 4)

- [ ] Deploy with Gunicorn + Nginx
- [ ] Set up CI/CD pipeline
- [ ] Add database for prediction logging
- [ ] Implement authentication/API keys
- [ ] Add rate limiting
- [ ] Create comprehensive API documentation (Swagger)
- [ ] Set up monitoring and alerting

## References

- [Flask Documentation](https://flask.palletsprojects.com/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [Bootstrap 5](https://getbootstrap.com/)
- [scikit-learn](https://scikit-learn.org/)

## Author
Machine Learning Team

## License
Proprietary - Brazilian Malware Dataset Project
