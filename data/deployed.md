# Deployed Application

## Live URL
**Production Deployment:** https://machine-learning-production-046b.up.railway.app

**Status:** ✅ Live and Operational

---

## Application Features

### 1. Single Instance Prediction
**Endpoint:** `/` (Home Page)

**Features:**
- Manual feature entry form with 26 input fields
- Pre-filled demo row option (click "Load Sample" buttons)
- Real-time prediction: Malware (1) or Goodware (0)
- Prediction confidence score displayed
- Supports both malware and goodware examples

**How to Use:**
1. Visit the live URL
2. Click "Load Malware Sample" or "Load Goodware Sample" for demo data
3. Or manually enter 26 feature values
4. Click "Predict" button
5. View classification result and confidence score

---

### 2. Batch CSV Upload
**Endpoint:** `/api/upload`

**Features:**
- Upload CSV file with multiple instances (up to 1000 rows)
- Automatic feature validation
- Batch predictions for all rows
- Results displayed in interactive table
- **Optional:** If CSV contains "Label" column, displays evaluation metrics:
  - AUC (Area Under the ROC Curve)
  - Accuracy
  - Confusion Matrix
  - Precision, Recall, F1-Score

**How to Use:**
1. Navigate to "Batch Upload" section on home page
2. Click "Choose File" and select CSV with 26 feature columns
3. Optional: Include "Label" column (0=goodware, 1=malware) for metrics
4. Click "Upload and Predict"
5. View predictions table and evaluation metrics (if labels provided)

**Sample File:** Test with the 20% hold-out test set for full metrics display

---

### 3. API Endpoints

#### Health Check
- **Endpoint:** `GET /health`
- **Response:** `{"status": "healthy", "model": "LightGBM", "version": "1.0"}`
- **Purpose:** Deployment monitoring and smoke tests

#### Get Features List
- **Endpoint:** `GET /api/features`
- **Response:** JSON array of 26 feature names
- **Purpose:** Client integration and form generation

#### Single Prediction
- **Endpoint:** `POST /api/predict`
- **Request Body:** JSON with 26 feature values
- **Response:** `{"prediction": 0/1, "probability": 0.0-1.0}`
- **Example:**
  ```json
  {
    "0": 0.5, "1": 0.3, ..., "25": 0.8
  }
  ```

#### Model Information
- **Endpoint:** `GET /api/model-info`
- **Response:** Model type, feature count, CV AUC, training date
- **Purpose:** Application metadata and documentation

#### Sample Data
- **Endpoints:** 
  - `GET /api/sample/goodware` — Load sample goodware instance
  - `GET /api/sample/malware` — Load sample malware instance
  - `GET /api/sample/random` — Load random sample
- **Response:** JSON with pre-filled feature values
- **Purpose:** Demo and testing

---

## Technical Specifications

### Hosting Platform
- **Provider:** Railway
- **Region:** Europe West (europe-west4)
- **Plan:** Free Tier (512MB RAM, 500MB storage)
- **Environment:** Python 3.11.9

### Technology Stack
- **Backend:** Flask 3.0.0
- **WSGI Server:** gunicorn 21.2.0 (1 worker, sync mode)
- **ML Framework:** LightGBM 4.1.0, scikit-learn 1.3.2
- **Data Processing:** pandas 2.1.3, numpy 1.26.2
- **Frontend:** HTML5, Bootstrap 5.3, JavaScript (vanilla)

### Build Configuration
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `cd Project/app && gunicorn --config gunicorn.conf.py app:app`
- **Port:** Dynamically assigned by Railway (binds to PORT environment variable)
- **Cold Start Time:** ~30 seconds (free tier)
- **Build Time:** ~2 minutes

### Performance
- **Model Load Time:** ~2 seconds on startup
- **Single Prediction:** <100ms response time
- **Batch Processing:** ~50-100ms per instance (depends on file size)
- **Maximum Upload Size:** 10MB (configurable)
- **Timeout:** 120 seconds for long-running requests

---

## CI/CD Pipeline

### Automated Deployment
- **Trigger:** Push to `main` branch on GitHub
- **Testing:** GitHub Actions runs 47 unit + integration tests
- **Deployment:** Automatic deployment to Railway ONLY if tests pass
- **Build Status:** Visible in GitHub Actions tab

### Workflow
```
1. Developer pushes code to main branch
   ↓
2. GitHub Actions triggers test workflow
   ↓
3. Tests run (pytest with 47 tests)
   ↓
4. If tests pass → Deploy to Railway
   ↓
5. Railway rebuilds and redeploys (2 min)
   ↓
6. Health check confirms deployment
   ↓
7. Live URL updated with new version
```

### Monitoring
- **Build Logs:** Available in Railway dashboard
- **Runtime Logs:** Accessible via Railway CLI or dashboard
- **Health Check:** `/health` endpoint monitored for uptime
- **Error Tracking:** Application logs include timestamps and error traces

---

## Usage Limits (Free Tier)

### Railway Free Tier Limits
- **Execution Time:** 500 hours/month (sufficient for demos and testing)
- **Memory:** 512MB RAM (adequate for LightGBM model)
- **Storage:** 500MB (model files ~550KB)
- **Bandwidth:** 100GB/month
- **Cold Starts:** Yes (~30s wake-up after inactivity)

### Recommendations
- **For Production:** Upgrade to Railway Starter ($5/month) for 0 cold starts
- **For High Traffic:** Consider Pro plan ($20/month) with autoscaling
- **For Demos:** Free tier is sufficient

---

## Security

### Data Privacy
- **No Data Storage:** Uploaded files processed in-memory, not persisted
- **No Logging of Inputs:** Feature values not logged (only predictions metadata)
- **HTTPS:** All traffic encrypted via Railway's SSL/TLS

### API Security
- **Rate Limiting:** None configured (free tier has natural throttling)
- **CORS:** Enabled for cross-origin requests
- **Input Validation:** All endpoints validate input format and feature count

### Recommendations for Production
- Add API key authentication
- Implement rate limiting (Flask-Limiter)
- Add request logging with PII redaction
- Set up monitoring/alerting (Sentry, DataDog)

---

## Testing the Deployment

### 1. Manual Testing
```bash
# Health check
curl https://machine-learning-production-046b.up.railway.app/health

# Single prediction (example)
curl -X POST https://machine-learning-production-046b.up.railway.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"0": 0.5, "1": 0.3, ..., "25": 0.8}'

# Get features list
curl https://machine-learning-production-046b.up.railway.app/api/features
```

### 2. UI Testing
1. Visit live URL in browser
2. Click "Load Malware Sample" → Predict → Verify result shows "Malware (1)"
3. Click "Load Goodware Sample" → Predict → Verify result shows "Goodware (0)"
4. Upload test CSV → Verify predictions table displays
5. Upload test set with labels → Verify metrics display (AUC, confusion matrix)

### 3. Automated Testing
```bash
# Run integration tests against live endpoint
pytest Project/app/tests/test_integration.py -v --url=https://machine-learning-production-046b.up.railway.app
```

---

## Troubleshooting

### Common Issues

**502 Bad Gateway:**
- **Cause:** App crashed or cold start timeout
- **Solution:** Wait 30s for cold start, refresh page

**File Upload Fails:**
- **Cause:** CSV missing required features or format incorrect
- **Solution:** Ensure 26 feature columns present (see `/api/features` for names)

**Predictions Seem Wrong:**
- **Cause:** Feature values not scaled/normalized correctly
- **Solution:** App handles preprocessing automatically; check input ranges

**Slow Response:**
- **Cause:** Cold start after inactivity
- **Solution:** Normal for free tier; first request takes ~30s, subsequent requests <1s

### Support
- **GitHub Issues:** [Repository Issues Page]
- **Email:** projects+msse@quantic.edu
- **Railway Status:** https://railway.app/status

---

## Deployment History

| Version | Date | Changes | Status |
|---------|------|---------|--------|
| 1.0 | Jan 18, 2026 | Initial production deployment with LightGBM model | ✅ Live |

---

## Future Enhancements

### Planned Features
- [ ] API authentication (JWT tokens)
- [ ] Request rate limiting
- [ ] Model versioning (A/B testing)
- [ ] Real-time monitoring dashboard
- [ ] Batch processing queue (Redis + Celery)
- [ ] Model retraining pipeline

### Infrastructure Upgrades
- [ ] Migrate to Railway Pro for 0 cold starts
- [ ] Add Redis for caching predictions
- [ ] Set up CDN for static assets
- [ ] Implement auto-scaling based on traffic

---

**Document Version:** 1.0  
**Last Updated:** January 18, 2026  
**Deployment:** Production (Railway Free Tier)  
**Uptime Target:** 99.5% (best effort on free tier)
