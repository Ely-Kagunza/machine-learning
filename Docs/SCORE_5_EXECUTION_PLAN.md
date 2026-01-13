# SCORE 5 EXECUTION PLAN
## Introduction to Machine Learning Project - Brazilian Malware Detection

**Goal:** Achieve a perfect score of 5 by delivering all requirements flawlessly.

**Timeline:** 5-6 weeks (solo) / 3-4 weeks (team of 3)

---

## SCORE 5 REQUIREMENTS CHECKLIST

### ✅ ML Models (Non-Negotiable)
- [ ] 4 Baseline models: Logistic Regression, Decision Tree, Random Forest, PyTorch MLP
- [ ] 3+ Additional models from different families: XGBoost, LightGBM, CatBoost (minimum)
- [ ] Complete CV evaluation: AUC and accuracy (mean ± std dev) for EACH model
- [ ] Complete test evaluation: Final hold-out test metrics for top model
- [ ] CV results table in report showing all models side-by-side with metrics

### ✅ Web Application (Fully Functional)
- [ ] UI Form: Manual feature entry with pre-filled demo row
- [ ] Predictions display: Shows malware/goodware classification
- [ ] File upload: Works reliably with multiple instances
- [ ] Evaluation metrics display: When test set uploaded, shows AUC, accuracy, confusion matrix
- [ ] No bugs or incomplete features — everything works smoothly

### ✅ Deployment (Flawless)
- [ ] Public URL: Live, accessible, no errors
- [ ] App responds correctly: Both single predictions and batch uploads work
- [ ] Hosting: Render, Railway, or Fly.io (proven free tiers)
- [ ] No deployment issues: Code deploys cleanly

### ✅ CI/CD Pipeline (Fully Functional)
- [ ] GitHub Actions configured
- [ ] Tests run automatically on PR or push to main
- [ ] Auto-deploy works: Deploys ONLY if tests pass
- [ ] Unit tests: Preprocessing functions, model wrapper class
- [ ] Integration tests: `/predict` endpoint with sample payload
- [ ] Smoke tests: `/health` endpoint confirms deployment
- [ ] Pipeline is visible/demonstrable in the video

### ✅ Documentation (Complete & Clear)
- [ ] evaluation-and-design.md: CV results table, final metrics, design decisions
- [ ] deployed.md: Live URL (tested and working)
- [ ] ai-tooling.md: Tools used, what worked, what didn't
- [ ] README.md: Clear setup/run instructions
- [ ] Code is reproducible: requirements.txt pinned, seeds set, train.py runs cleanly

### ✅ Demo Presentation (5-10 min, Professional)
- [ ] Shows web app functionality (manual entry, file upload, metrics)
- [ ] Shows CI/CD pipeline (commit → test → deploy)
- [ ] All group members present & speaking
- [ ] Clear audio/video quality
- [ ] Professional delivery

---

## WEEK 1: FOUNDATION (DO THIS RIGHT)

### Day 1-2: Setup & Repository Structure

**Objectives:**
- Create professional GitHub repo with proper structure
- Set up virtual environment with pinned dependencies
- Establish reproducibility from day one

**Tasks:**

1. **Create GitHub Repository**
   - Initialize with README, .gitignore (Python)
   - Add "quantic-grader" as collaborator immediately
   - Clone locally

2. **Create Project Structure**
   ```
   project/
   ├── src/
   │   ├── preprocessing.py       # Preprocessing pipeline
   │   ├── model_wrapper.py       # Model loading/inference
   │   ├── features.py            # Feature handling
   │   └── utils.py               # Utility functions
   ├── models/
   │   ├── trained_model.pkl      # Best model saved here
   │   └── preprocessor.pkl       # Fitted preprocessor
   ├── tests/
   │   ├── test_preprocessing.py
   │   ├── test_models.py
   │   └── test_api.py
   ├── data/
   │   ├── train.csv              # 80% split
   │   └── test.csv               # 20% split
   ├── notebooks/
   │   └── eda.ipynb              # Exploratory analysis
   ├── app.py                      # Flask application
   ├── train.py                    # Training script
   ├── eval.py                     # Evaluation script
   ├── requirements.txt            # Pinned dependencies
   └── README.md
   ```

3. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
   pip install --upgrade pip
   ```

4. **Create requirements.txt (Pinned Versions)**
   ```
   pandas==2.0.3
   numpy==1.24.3
   scikit-learn==1.3.0
   torch==2.0.1
   xgboost==2.0.0
   lightgbm==4.0.0
   catboost==1.2.1
   flask==3.0.0
   pytest==7.4.0
   pytest-cov==4.1.0
   joblib==1.3.1
   matplotlib==3.7.2
   seaborn==0.12.2
   ```

5. **Initial Git Commit**
   ```bash
   git add .
   git commit -m "Initial project setup with directory structure"
   ```

**Checklist:**
- [ ] GitHub repo created and quantic-grader added
- [ ] Project structure created
- [ ] Virtual environment activated
- [ ] requirements.txt created with pinned versions
- [ ] Initial commit pushed

---

### Day 3: Data Split & EDA Preparation

**Objectives:**
- Split data 80/20 BEFORE any preprocessing (prevent leakage)
- Understand dataset structure and characteristics
- Document findings

**Tasks:**

1. **Load and Explore Dataset**
   - Load `brazilian-malware.csv` and `goodware.csv`
   - Combine into single dataset
   - Check shape, data types, missing values
   - Verify Label column (0=goodware, 1=malware)

2. **Perform 80/20 Train/Test Split**
   - Use `train_test_split` with `stratify=y` (maintain class balance)
   - Set `random_state=42` for reproducibility
   - Save to `data/train.csv` and `data/test.csv`
   - **DO NOT touch test.csv again until final evaluation**

3. **Document Split**
   - Record split sizes (e.g., 40,000 train, 10,000 test)
   - Record class distribution in train/test
   - Ensure balanced splits

4. **Start EDA Notebook**
   - Create `notebooks/eda.ipynb`
   - Load training data ONLY
   - Explore feature distributions (histogram, box plots)
   - Check for missing values, outliers
   - Analyze class balance
   - Document findings

**Checklist:**
- [ ] Train/test split created with stratification
- [ ] data/train.csv and data/test.csv saved
- [ ] Class distribution recorded
- [ ] EDA notebook started
- [ ] Commit pushed: "Complete data split and begin EDA"

---

### Days 4-5: Preprocessing Pipeline & Reproducibility

**Objectives:**
- Build preprocessing pipeline that fits on train only
- Create reproducible training script
- Document preprocessing decisions

**Tasks:**

1. **Build Preprocessing Pipeline** (`src/preprocessing.py`)
   - Handle missing values (imputation strategy)
   - Scale numerical features (StandardScaler)
   - Encode categorical features if needed
   - **Critical:** Use `Pipeline` from scikit-learn
   - Fit ONLY on training data

2. **Create Model Wrapper** (`src/model_wrapper.py`)
   - Class to encapsulate preprocessing + model
   - Load/save functionality
   - Inference method

3. **Create Training Script** (`train.py`)
   - Load train.csv
   - Apply preprocessing
   - Save preprocessor (joblib)
   - Ready for model training next week
   - All with fixed random seeds

4. **Document Preprocessing Decisions**
   - Why choose this imputation strategy?
   - Why this scaling method?
   - Any features dropped? Why?
   - Save in preparation for evaluation-and-design.md

**Checklist:**
- [ ] Preprocessing pipeline created (fits on train only)
- [ ] Model wrapper class created
- [ ] train.py script functional
- [ ] Preprocessing documented
- [ ] Commit pushed: "Build preprocessing pipeline"

---

## WEEK 2: ML MODELS (HIGH QUALITY)

### Days 1-2: Baseline Models with Cross-Validation

**Objectives:**
- Implement 4 baseline models
- Set up 10-fold stratified cross-validation
- Begin result tracking

**Tasks:**

1. **Implement 10-Fold Stratified Cross-Validation**
   - Use `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`
   - Create validation script to evaluate all models consistently

2. **Train Baseline Models**
   - **Logistic Regression:** Default, record CV metrics
   - **Decision Tree:** Tune `max_depth`, `min_samples_split` via CV grid search
   - **Random Forest:** Tune `n_estimators`, `max_depth` via CV grid search
   - **PyTorch MLP:** Build simple architecture, train with CV (challenging—use AI for scaffolding)

3. **Record Results**
   - For each model: CV AUC (mean ± std), CV Accuracy (mean ± std)
   - Create results table as you go
   - Save model objects for reference

4. **Use AI Tools**
   - Use ChatGPT/Copilot for PyTorch MLP boilerplate
   - Use for hyperparameter grid setup
   - Focus on understanding, not copying blindly

**Checklist:**
- [ ] 10-fold stratified CV set up
- [ ] Logistic Regression trained and evaluated
- [ ] Decision Tree trained with hyperparameter tuning
- [ ] Random Forest trained with hyperparameter tuning
- [ ] PyTorch MLP trained and evaluated
- [ ] Results table created with CV metrics
- [ ] Commit pushed: "Implement baseline models with CV"

---

### Days 3-4: Advanced Models

**Objectives:**
- Train 3+ additional models from different families
- Hyperparameter tuning for each
- Expand results table

**Tasks:**

1. **Train XGBoost Model**
   - Hyperparameter grid: learning_rate, max_depth, n_estimators
   - CV evaluation, record AUC ± std, Accuracy ± std

2. **Train LightGBM Model**
   - Hyperparameter grid: learning_rate, num_leaves, n_estimators
   - CV evaluation, record metrics

3. **Train CatBoost Model**
   - Hyperparameter grid: learning_rate, depth, iterations
   - CV evaluation, record metrics

4. **Consider 4th Advanced Model** (Optional for polish)
   - SVM with RBF kernel
   - Gradient Boosting Classifier
   - Or stacking ensemble of top models

5. **Update Results Table**
   - All 7+ models in one table
   - Format: Model Name | AUC (mean ± std) | Accuracy (mean ± std)

**Checklist:**
- [ ] XGBoost trained with CV
- [ ] LightGBM trained with CV
- [ ] CatBoost trained with CV
- [ ] Additional model (if pursuing)
- [ ] Comprehensive results table created
- [ ] Commit pushed: "Add advanced models with hyperparameter tuning"

---

### Day 5: Model Selection & Final Evaluation

**Objectives:**
- Select top-performing model from CV results
- Evaluate on hold-out test set
- Prepare for web app

**Tasks:**

1. **Analyze CV Results**
   - Compare all models by AUC (primary metric)
   - Note which performed best
   - Document reasoning for selection

2. **Final Evaluation on Test Set**
   - Load test.csv (FIRST TIME USING IT)
   - Apply preprocessing (transform only, don't fit)
   - Generate predictions with selected model
   - Calculate test AUC, test Accuracy, Confusion Matrix

3. **Save Selected Model**
   - Pickle best model to `models/trained_model.pkl`
   - Save preprocessor to `models/preprocessor.pkl`
   - Record model name, test metrics

4. **Create Evaluation Document Draft**
   - Will be expanded later for evaluation-and-design.md
   - Include CV table, test results, selection reasoning

5. **Create eval.py Script**
   - Loads test set
   - Loads best model
   - Generates evaluation metrics
   - Outputs confusion matrix, AUC plots

**Checklist:**
- [ ] Top model selected based on CV AUC
- [ ] Final evaluation on test set completed
- [ ] Test AUC, Accuracy recorded
- [ ] Confusion matrix generated
- [ ] Best model saved (pkl format)
- [ ] eval.py script created
- [ ] Commit pushed: "Complete model selection and test evaluation"

---

## WEEK 3: WEB APPLICATION (PROFESSIONAL)

### Days 1-2: Flask Setup & Basic Endpoints

**Objectives:**
- Create Flask app structure
- Load trained model
- Implement /predict endpoint

**Tasks:**

1. **Create Flask App Structure** (`app.py`)
   ```python
   from flask import Flask, request, jsonify, render_template
   import joblib
   import pandas as pd
   
   app = Flask(__name__)
   
   # Load model and preprocessor
   model = joblib.load('models/trained_model.pkl')
   preprocessor = joblib.load('models/preprocessor.pkl')
   ```

2. **Implement /predict Endpoint (Single Instance)**
   - Accept POST request with JSON feature data
   - Apply preprocessing
   - Generate prediction (malware=1 or goodware=0)
   - Return JSON response with prediction and confidence

3. **Implement /health Endpoint**
   - Simple GET endpoint
   - Returns `{"status": "healthy"}`
   - Used for smoke tests and deployment verification

4. **Test Locally**
   - Use curl or Postman to test /predict
   - Test with sample malware instance
   - Test with sample goodware instance
   - Verify /health works

5. **Create Demo Dataset Entry**
   - Extract one row from test.csv (malware example)
   - Use as pre-fill default in form
   - Saves users from entering 27 features manually

**Checklist:**
- [ ] Flask app created with app.py
- [ ] Model and preprocessor loaded correctly
- [ ] /predict endpoint implemented and tested
- [ ] /health endpoint implemented
- [ ] Local testing successful
- [ ] Commit pushed: "Implement Flask app with /predict endpoint"

---

### Days 2-3: UI Form & File Upload

**Objectives:**
- Create professional HTML form
- Implement file upload endpoint
- Display predictions and metrics

**Tasks:**

1. **Create HTML Templates** (`templates/`)
   - `index.html`: Main page with form and results
   - Form with 27 feature inputs (auto-populated with demo row)
   - Submit button triggers /predict
   - Display area for prediction result

2. **Implement File Upload Endpoint** (`/upload` or `/batch-predict`)
   - Accept CSV file upload
   - Parse features from CSV
   - Generate predictions for all rows
   - Return predictions in table format

3. **Implement Metrics Display**
   - If uploaded file contains "Label" column:
     - Calculate AUC, Accuracy
     - Generate Confusion Matrix
     - Display metrics on page
   - Can demo with test.csv to show model performance

4. **HTML/CSS Polish**
   - Clean, professional UI
   - Clear labeling of fields
   - Error messages for invalid uploads
   - Results displayed clearly
   - Mobile-friendly if possible

5. **Test Thoroughly**
   - Manual feature entry → prediction works
   - File upload → predictions display
   - Test with malware and goodware examples
   - Upload test.csv → metrics display correctly

**Checklist:**
- [ ] HTML form created with pre-filled demo row
- [ ] Manual prediction working end-to-end
- [ ] File upload endpoint implemented
- [ ] Metrics calculation working
- [ ] UI is clean and professional
- [ ] All features tested locally
- [ ] Commit pushed: "Add UI form and file upload functionality"

---

### Days 4: Edge Cases & Error Handling

**Objectives:**
- Make app production-ready
- Handle edge cases gracefully
- Ensure no crashes

**Tasks:**

1. **Error Handling**
   - Missing features in input → return error message
   - Invalid CSV format → return error message
   - File too large → reject gracefully
   - Invalid feature values → return error message

2. **Input Validation**
   - Check all 27 features present
   - Check feature value ranges
   - Sanitize user inputs

3. **Edge Case Testing**
   - Upload empty file
   - Upload with wrong columns
   - Submit form with missing fields
   - Submit with invalid numbers
   - Ensure app never crashes

4. **Logging**
   - Add basic logging for debugging
   - Log prediction requests (helpful for testing)

**Checklist:**
- [ ] Error handling for all common issues
- [ ] Input validation implemented
- [ ] Edge cases tested
- [ ] No crashes or 500 errors
- [ ] Logging in place
- [ ] Commit pushed: "Add error handling and input validation"

---

## WEEK 4: TESTING & DEPLOYMENT

### Day 1: Unit & Integration Tests

**Objectives:**
- Comprehensive test suite
- All components tested
- CI/CD ready

**Tasks:**

1. **Unit Tests** (`tests/test_preprocessing.py`)
   ```python
   def test_preprocessing_shapes():
       # Test preprocessor output shape
       pass
   
   def test_preprocessing_no_nans():
       # Test no NaN in output
       pass
   
   def test_scaling():
       # Test features are scaled correctly
       pass
   ```

2. **Model Wrapper Tests** (`tests/test_models.py`)
   ```python
   def test_model_prediction_shape():
       # Test prediction output shape
       pass
   
   def test_model_malware_prediction():
       # Test malware prediction works
       pass
   
   def test_model_goodware_prediction():
       # Test goodware prediction works
       pass
   ```

3. **API Integration Tests** (`tests/test_api.py`)
   ```python
   def test_predict_endpoint():
       # Test /predict with valid JSON payload
       response = client.post('/predict', json=test_features)
       assert response.status_code == 200
       assert 'prediction' in response.json
   
   def test_health_endpoint():
       # Test /health
       response = client.get('/health')
       assert response.status_code == 200
       assert response.json['status'] == 'healthy'
   
   def test_upload_endpoint():
       # Test /upload with CSV file
       pass
   
   def test_invalid_input():
       # Test error handling
       pass
   ```

4. **Run Tests Locally**
   ```bash
   pytest tests/ -v --cov=src
   ```

5. **Ensure All Tests Pass**
   - Fix any failures
   - Aim for >80% code coverage

**Checklist:**
- [ ] Unit tests for preprocessing written and passing
- [ ] Unit tests for model wrapper written and passing
- [ ] Integration tests for API endpoints written and passing
- [ ] Tests for error handling written and passing
- [ ] All tests passing locally
- [ ] Code coverage >80%
- [ ] Commit pushed: "Add comprehensive test suite"

---

### Days 2-3: GitHub Actions CI/CD Pipeline

**Objectives:**
- Automated testing on PR/push
- Automated deployment on test pass
- Pipeline fully functional

**Tasks:**

1. **Create GitHub Actions Workflow** (`.github/workflows/test-deploy.yml`)
   ```yaml
   name: Test and Deploy
   
   on:
     push:
       branches: [main]
     pull_request:
       branches: [main]
   
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - uses: actions/setup-python@v4
           with:
             python-version: '3.10'
         - run: pip install -r requirements.txt
         - run: pytest tests/ -v
   
     deploy:
       needs: test
       runs-on: ubuntu-latest
       if: github.ref == 'refs/heads/main' && github.event_name == 'push'
       steps:
         - uses: actions/checkout@v3
         - name: Deploy to [Render/Railway]
           # Deployment step (specific to platform)
   ```

2. **Configure Deployment Secrets**
   - Add deployment credentials to GitHub Secrets
   - (Render API key, Railway API key, etc.)

3. **Test Pipeline Locally** (optional)
   - Make a test PR
   - Watch workflow run
   - Verify tests execute
   - Merge and verify auto-deploy

4. **Document Pipeline**
   - Comment in workflow file
   - Explain each step
   - Will be referenced in demo

**Checklist:**
- [ ] GitHub Actions workflow created
- [ ] Tests run automatically on PR/push
- [ ] Deployment secrets configured
- [ ] Pipeline tested with test PR
- [ ] Auto-deploy verified
- [ ] Workflow documented
- [ ] Commit pushed: "Add GitHub Actions CI/CD pipeline"

---

### Days 3-4: Deploy to Production (Render or Railway)

**Objectives:**
- App live and accessible
- Auto-deploy working
- Public URL ready for demo

**Tasks:**

1. **Choose Hosting Platform**
   - **Render** (easiest): Free tier, easy GitHub integration
   - **Railway**: Also free tier, slightly more flexible

2. **Create Render Account & Connect GitHub**
   - Sign up at render.com
   - Connect GitHub repo
   - Create new Web Service
   - Select repo and branch (main)

3. **Configure Deployment**
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app` (requires gunicorn in requirements.txt)
   - Environment variables: Any needed configs
   - Add gunicorn to requirements.txt: `gunicorn==21.2.0`

4. **Deploy First Time Manually**
   - Test deployment works
   - Verify public URL accessible
   - Test /health endpoint on live site
   - Test /predict endpoint with sample data

5. **Connect CI/CD to Deployment**
   - Configure GitHub Actions to trigger Render deployment on test pass
   - Or: Render auto-deploys on GitHub push (simplest option)
   - Test: Make code change, push to main, watch auto-deploy

6. **Verify Live App**
   - Visit public URL
   - Test form submission
   - Test file upload
   - Ensure all features work on live site

**Checklist:**
- [ ] Hosting platform chosen (Render or Railway)
- [ ] Account created and GitHub connected
- [ ] gunicorn added to requirements.txt
- [ ] Web service created
- [ ] First manual deployment successful
- [ ] Public URL working and accessible
- [ ] All endpoints tested on live site
- [ ] Auto-deploy configured and tested
- [ ] Deployed URL documented for deployed.md
- [ ] Commit pushed: "Deploy to [platform]"

---

## WEEK 5: DOCUMENTATION & DEMO

### Days 1-2: Write Documentation Files

**Objectives:**
- Complete documentation
- All decisions explained
- Reproducibility guaranteed

**Tasks:**

1. **Create evaluation-and-design.md**
   ```markdown
   # Evaluation and Design Decisions
   
   ## Cross-Validation Results
   | Model | AUC (mean ± std) | Accuracy (mean ± std) |
   |-------|------------------|----------------------|
   | ... (all 7+ models) |
   
   ## Final Test Set Evaluation
   - Selected Model: [Name]
   - Test AUC: [Value]
   - Test Accuracy: [Value]
   - Confusion Matrix: [Values]
   
   ## Design Decisions
   ### Data Preprocessing
   - Imputation strategy: [Why chosen]
   - Scaling method: [Why StandardScaler]
   - Features dropped/engineered: [Why]
   
   ### Model Selection
   - Why was [Model] chosen over others?
   - Cross-validation approach and rationale
   - Hyperparameter tuning decisions
   
   ### Data Leakage Prevention
   - Train/test split done BEFORE preprocessing
   - Preprocessing fit only on training folds
   ```

2. **Create deployed.md**
   ```markdown
   # Deployed Application
   
   ## Live URL
   [https://your-app-name.onrender.com](https://your-app-name.onrender.com)
   
   ## Features
   - Manual prediction form
   - Batch file upload
   - Evaluation metrics display
   ```

3. **Create ai-tooling.md**
   ```markdown
   # AI Tools Used
   
   ## Tools & Usage
   - **ChatGPT/Copilot**: Used for Flask boilerplate scaffolding
     - What worked: Quick setup of Flask structure
     - What didn't: Generic code needed customization for our needs
   
   - **GitHub Copilot**: Used for test writing
     - What worked: Suggested good test patterns
     - What didn't: Didn't understand our custom pipeline
   
   ## Evaluation
   - AI tools saved ~20 hours on boilerplate
   - Still required significant customization and debugging
   - Logic and decision-making remained human-driven
   ```

4. **Create/Update README.md**
   ```markdown
   # Brazilian Malware Detection ML Project
   
   ## Setup
   ```bash
   python -m venv venv
   source venv/Scripts/activate
   pip install -r requirements.txt
   ```
   
   ## Training
   ```bash
   python train.py
   ```
   
   ## Evaluation
   ```bash
   python eval.py
   ```
   
   ## Run Web App
   ```bash
   flask run
   ```
   
   ## Testing
   ```bash
   pytest tests/ -v
   ```
   ```

5. **Code Comments**
   - Add docstrings to key functions
   - Comment non-obvious logic
   - Explain preprocessing steps

**Checklist:**
- [ ] evaluation-and-design.md complete with CV table and test results
- [ ] deployed.md with live URL
- [ ] ai-tooling.md documenting tool usage
- [ ] README.md with clear instructions
- [ ] Code well-commented
- [ ] All documentation proofread
- [ ] Commit pushed: "Add comprehensive documentation"

---

### Days 3-4: Record Demo Presentation

**Objectives:**
- Professional 5-10 minute video
- Show all functionality
- Explain design and process

**Tasks:**

1. **Script the Demo** (5-10 minutes)
   ```
   0:00-0:30   - Introduction & project overview
   0:30-2:00   - Show web app: manual prediction form
                 Enter features, click predict, show result
   2:00-3:00   - Show file upload: upload test.csv
                 Display batch predictions table
   3:00-4:00   - Show metrics: AUC, accuracy, confusion matrix
   4:00-5:00   - Show GitHub repo & code structure
   5:00-7:00   - Show CI/CD pipeline:
                 Push code, watch GitHub Actions run tests,
                 Watch auto-deploy to Render
   7:00-7:30   - Show live deployed app working
   7:30-8:00   - Summary & questions
   ```

2. **Prepare Screen Share Setup**
   - Open browser to live app URL
   - Have GitHub repo ready
   - Have GitHub Actions workflow ready
   - Have test.csv ready for upload demo

3. **Record Demo**
   - Use screen recording tool (OBS, ScreenFlow, built-in recording)
   - Ensure audio is clear
   - Speak clearly and confidently
   - Practice beforehand 2-3 times
   - Multiple takes if needed

4. **Team Coordination** (if team)
   - Divide speaking parts among members
   - Ensure all members speak and appear on camera
   - Everyone is present throughout video
   - Practice transitions between speakers

5. **Video Requirements**
   - Length: 5-10 minutes
   - Format: MP4 or common video format
   - Audio: Clear, no background noise
   - Video: HD if possible (1080p)
   - Upload to: YouTube (unlisted) or Google Drive

**Checklist:**
- [ ] Demo script written
- [ ] All team members have speaking parts
- [ ] Screen recording tested and working
- [ ] Live app tested and working before recording
- [ ] Demo recorded (multiple takes, best one selected)
- [ ] Video uploaded to YouTube (unlisted) or Google Drive
- [ ] Video link ready for submission PDF

---

### Day 5: Final Review & Submission Preparation

**Objectives:**
- Everything works
- Ready for grading
- Submission package complete

**Tasks:**

1. **Comprehensive Checklist**
   - [ ] All code in GitHub repo
   - [ ] All documentation complete and proofread
   - [ ] quantic-grader is collaborator
   - [ ] Live URL tested and working
   - [ ] All tests passing
   - [ ] CI/CD pipeline visible on GitHub
   - [ ] train.py reproduces results
   - [ ] eval.py works correctly
   - [ ] Flask app fully functional
   - [ ] Demo video ready

2. **Verify Reproducibility**
   ```bash
   # Fresh clone in new directory
   git clone [your-repo]
   cd [repo]
   python -m venv venv
   source venv/Scripts/activate
   pip install -r requirements.txt
   python train.py
   python eval.py
   ```
   - Should run without errors
   - Results should match your documentation

3. **Test Live App One More Time**
   - Visit public URL
   - Test manual prediction
   - Test file upload
   - Test with test.csv (verify metrics display)
   - Test /health endpoint

4. **Prepare Submission PDF**
   - Create PDF document with:
     - **Link 1:** Recorded demo video URL (YouTube unlisted link)
     - **Link 2:** GitHub repository URL
   - Include brief intro explaining links
   - Keep PDF simple and professional

5. **Final Git Push**
   ```bash
   git add .
   git commit -m "Final submission: Complete ML project with Flask app, CI/CD, and full documentation"
   git push origin main
   ```

6. **Submit**
   - Go to dashboard
   - Click "Submit Project"
   - Upload PDF with links
   - Confirm submission

**Checklist:**
- [ ] All code committed and pushed
- [ ] Reproducibility verified (fresh clone test)
- [ ] Live app tested thoroughly
- [ ] Demo video uploaded and link ready
- [ ] Submission PDF created with links
- [ ] Final push to GitHub completed
- [ ] Project submitted through dashboard

---

## CRITICAL SUCCESS FACTORS

### ✅ Data Integrity
- Train/test split BEFORE preprocessing ✓
- Test set untouched until final evaluation ✓
- Stratified splits to maintain class balance ✓

### ✅ Reproducibility
- requirements.txt with pinned versions ✓
- Fixed random seeds (random_state=42) ✓
- train.py script that runs cleanly ✓
- Anyone can clone and reproduce results ✓

### ✅ Model Quality
- All 7+ models evaluated with CV ✓
- Results table shows mean ± std dev ✓
- Final test evaluation on hold-out set ✓
- Selection reasoning documented ✓

### ✅ Web App Quality
- No bugs or crashes ✓
- All features work smoothly ✓
- Error handling for edge cases ✓
- Professional UI ✓
- Live and publicly accessible ✓

### ✅ Testing & Deployment
- Comprehensive unit + integration tests ✓
- CI/CD pipeline fully automated ✓
- Tests run before deployment ✓
- Only deploys if tests pass ✓

### ✅ Documentation
- CV results table complete ✓
- Design decisions explained ✓
- AI tool usage documented ✓
- README with setup instructions ✓

### ✅ Demo Presentation
- All functionality shown ✓
- CI/CD pipeline demonstrated ✓
- All team members speak ✓
- Professional delivery ✓

---

## WHAT KILLS SCORE 5

❌ Data leakage (preprocessing before split)
❌ Only a few models instead of 7+
❌ Missing CV results or test evaluation
❌ Web app with bugs or missing features
❌ CI/CD configured but not working
❌ Deployment issues or live app broken
❌ Incomplete documentation
❌ Demo doesn't show all functionality
❌ Code not reproducible
❌ Sloppy presentation

---

## TIMELINE AT A GLANCE

| Week | Focus | Key Deliverable |
|------|-------|-----------------|
| 1 | Foundation & Setup | Train/test split, EDA, preprocessing pipeline |
| 2 | ML Models | 7+ models trained, CV results table, test evaluation |
| 3 | Web Application | Flask app functional, file upload, metrics display |
| 4 | Testing & Deployment | Tests passing, CI/CD working, app live on web |
| 5 | Documentation & Demo | All docs complete, video recorded, ready to submit |

---

## TIPS FOR SUCCESS

1. **Start immediately** — The ML modeling phase takes longest
2. **Use AI tools early** — Save time on boilerplate, focus on logic
3. **Test continuously** — Don't discover bugs on submission day
4. **Keep data clean** — No leakage, proper splits, reproducible seeds
5. **Document as you go** — Don't leave docs to last week
6. **Practice the demo** — Run through multiple times before recording
7. **Commit frequently** — Small commits with clear messages help debugging
8. **Pair programming** (if team) — Better code, shared understanding
9. **Verify reproducibility** — Fresh clone + fresh setup = real test
10. **Aim for the rubric** — Check each requirement as you go

---

## CONTINGENCY PLANNING

**If you're behind schedule:**
- Week 2: Focus on baseline models first (drop advanced if needed, add later)
- Week 3: Simple Flask app with basic UI (advanced features secondary)
- Week 4: Basic tests, minimal CI/CD (demo the concept, not production-grade)
- Week 5: Documentation and demo (non-negotiable)

**If a component fails:**
- Model underperforming? Add more models, ensemble them
- Flask app buggy? Roll back, simplify features
- Deployment failing? Use Railway instead of Render
- Tests failing? Fix highest priority tests first

---

## RESOURCES & REFERENCES

- **Scikit-learn:** [https://scikit-learn.org](https://scikit-learn.org)
- **PyTorch:** [https://pytorch.org](https://pytorch.org)
- **XGBoost/LightGBM/CatBoost:** Official docs
- **Flask:** [https://flask.palletsprojects.com](https://flask.palletsprojects.com)
- **GitHub Actions:** [https://docs.github.com/actions](https://docs.github.com/actions)
- **Render Deployment:** [https://render.com](https://render.com)
- **pytest:** [https://docs.pytest.org](https://docs.pytest.org)

---

**Version:** 1.0
**Date Created:** January 13, 2026
**Status:** Ready to Execute
