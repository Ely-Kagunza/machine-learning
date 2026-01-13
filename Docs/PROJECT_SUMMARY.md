# Complete Project Summary: Introduction to Machine Learning

## PROJECT OVERVIEW

**Objective:** Build and evaluate ML models to classify software as malware or goodware, deploy the best model in a web application, and implement a CI/CD pipeline with automated testing.

**Team Size:** Individual or up to 3 people

**Dataset:** Brazilian Malware Dataset (~50,000 instances, 27 input features, target variable in Column L: 0=goodware, 1=malware)

---

## LEARNING OUTCOMES

Upon completion, you will be able to:
- Preprocess and analyze structured data correctly
- Train, tune, and evaluate ML models using cross-validation and test sets
- Package trained ML models for production inference
- Build and deploy web applications serving ML predictions
- Implement CI/CD pipelines with automated testing
- Effectively demonstrate and communicate ML results

---

## PROJECT DESCRIPTION

**Problem Type:** Machine learning-based **static malware detection** — predicting if an executable (.exe) file is malicious before execution based on static file features.

**Alternative Options:** You can propose a different dataset and predictive problem (email: projects+msse@quantic.edu for approval).

**Tools Required:** Python with scikit-learn, PyTorch, and other ML/data manipulation libraries. **AI code generation tools are encouraged** — you'll be graded on quality and functionality, not hand-coding percentage. Document your AI tool usage.

---

## STEP-BY-STEP INSTRUCTIONS

### 1. Dataset & Problem Definition
- Confirm dataset and target variable
- Define success metrics:
  - **Primary:** AUC (Area Under Curve)
  - **Secondary:** Accuracy
- **Critical:** Hold out 20% test set BEFORE any preprocessing to prevent data leakage

### 2. Environment & Reproducibility
- Use virtual environment (venv, conda, etc.)
- Pin all dependencies in `requirements.txt` or `environment.yml`
- Provide reproducible scripts (`train.py`, `eval.py`)
- Set fixed random seeds for splits and cross-validation

### 3. Data Understanding & Preparation
- Conduct **Exploratory Data Analysis (EDA):**
  - Inspect feature distributions
  - Check class balance
  - Identify missing data
- Document issues and how they're handled
- Apply preprocessing ONLY on training data

### 4. Train/Validation/Test Protocol
- **80% training / 20% hold-out test split** (stratified by class)
- Perform **stratified 10-fold cross-validation** within training set for model selection and hyperparameter tuning
- Keep test set untouched until final evaluation

### 5. Preprocessing & Feature Engineering
- Apply scaling, encoding, imputation as needed
- **Critical:** Fit transformations on training folds ONLY; apply to validation/test folds
- Explore feature selection or dimensionality reduction (justify choices)

### 6. Model Training & Evaluation
**Required Baseline Models:**
- Logistic Regression
- Decision Tree
- Random Forest
- PyTorch Multi-Layer Perceptron (MLP)

**Additional Models:** Evaluate at least 3 more high-performing models spanning ≥2 different algorithm families (e.g., XGBoost, LightGBM, CatBoost)

**Reporting:**
- Record CV results: AUC and accuracy (mean ± std dev) for all models
- Select top-performing model from CV for final test set evaluation
- Report final production model performance on hold-out test set

### 7. Web Application Development (Flask)
Build a web app with:
- **UI form:** Manual feature entry with pre-filled demo row option; displays prediction (malware/goodware)
- **File upload:** Batch predictions for multiple instances
- **Evaluation display:** If uploaded file contains labels, show AUC, accuracy, and confusion matrix (can demo with 20% test set)

### 8. Web Application Deployment
- Deploy to free-tier hosting (Render, Railway, Fly.io, etc.)
- Application must be publicly accessible via shareable URL

### 9. CI/CD Pipeline
- Implement basic pipeline (e.g., GitHub Actions)
- **Minimum requirement:** Automated tests before deployment
  - Trigger on pull requests (before merge) OR pushes to main branch
- **Deployment:** Only deploy if tests pass
  - If push to main fails tests: code stays on main but doesn't deploy

### 10. Automated Testing
- **Unit tests:** Preprocessing and model wrapper functions
- **Integration tests:** Check `/predict` API endpoint with sample payload
- **Smoke tests:** GET `/health` endpoint to confirm deployment

---

## SUBMISSION GUIDELINES

**Format:** Single PDF containing:
1. Link to recorded demo presentation
2. Link to GitHub repository

**GitHub Repository Requirements:**
- Add "quantic-grader" as collaborator (Settings > Collaborators > Add people)
- Include all source code and CI/CD configuration
- **deployed.md file:** URL to live deployed web application
- **evaluation-and-design.md file:** 
  - CV results table for all models
  - Final hold-out test set evaluation
  - Explanation of design decisions, preprocessing, feature engineering
- **ai-tooling.md file:** Brief description of AI tools used for code generation/engineering, what worked well, what didn't
- **Recorded demo video (5-10 minutes):**
  - Screen-share showing web app at public URL
  - Demonstrate inference/predictions
  - Show CI/CD pipeline operation and automated testing
  - All group members must speak and appear on camera
  - No ID required

**Submission Process:** Click "Submit Project" on dashboard (one member submits for group)

**Questions?** Email: msse+projects@quantic.edu

**Grading Timeline:** 3-4 weeks after due date (no penalty for late submission, but grading may be delayed)

---

## PLAGIARISM POLICY

**Definition:** "Knowingly representing the work of others as one's own, engaging in any acts of plagiarism, or referencing works without appropriate citation."

**Key Points:**
- Quantic monitors all submissions for plagiarism
- All plagiarism (intentional or unintentional) is a conduct violation
- When in doubt, cite sources
- Academic integrity is essential; projects must be completed with your own efforts

---

## PROJECT RUBRIC (Score 2+ = Passing)

| Score | Criteria |
|-------|----------|
| **5** (Excellent) | ✓ ALL requirements met: Baseline + ≥3 additional models with complete CV/test evaluation ✓ CV table (AUC/accuracy ± SD) + final test metrics in report ✓ Fully functional web app with UI, file upload, metrics display ✓ Successful public deployment ✓ Fully functional CI/CD with tests on PR/push and auto-deploy ✓ Substantial unit + integration + smoke tests ✓ Clear, effective demo showing UI + CI/CD operation |
| **4** (Good) | ✓ MOST requirements met ✓ All baselines + ≥3 additional models (minor omissions) ✓ Mostly reproducible environment (small gaps) ✓ CV table + test metrics (minor clarity issues) ✓ Functional web app (some UI/upload features incomplete) ✓ Public deployment with minor issues ✓ CI/CD present (some test/deploy limitations) ✓ Demo covers most aspects |
| **3** (Satisfactory) | ✓ SOME requirements met ✓ Four baselines trained; <3 additional models OR shallow evaluation ✓ Partial reproducibility ✓ CV results (lacks depth/completeness) ✓ Partial web app (test upload missing/incomplete) ✓ Deployment with issues ✓ Incomplete CI/CD ✓ Unclear or missing functionality demo |
| **2** (Minimal Pass) | ✓ FEW requirements met ✓ <4 baselines; minimal additional models ✓ Poor reproducibility ✓ Sparse report, missing results tables ✓ Minimal/non-functional web app ✓ Significant deployment issues ✓ Lacking/minimal CI/CD ✓ Weak/limited demo |
| **1** (Fail - Needs Revision) | ✗ MOST requirements missing ✗ Baselines not implemented correctly ✗ No reproducibility ✗ Report largely absent ✗ No working web app ✗ No deployment ✗ No CI/CD/tests ✗ No demo |
| **0** (No Credit) | ✗ Assignment not completed, plagiarized, or completely fails to address requirements |

**Note:** Must revise and resubmit to receive credit if scoring 1 or 0.
