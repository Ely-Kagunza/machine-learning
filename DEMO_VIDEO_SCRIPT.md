# Demo Video Script & Recording Guide

## Video Requirements
- **Duration:** 5-10 minutes
- **Format:** MP4 or common video format
- **Quality:** HD (1080p preferred)
- **Audio:** Clear, no background noise
- **Requirement:** You must appear on camera and speak throughout

---

## VIDEO SCRIPT (5-10 minutes)

### **0:00-0:30 — Introduction (30 seconds)**

**What to say:**
> "Hello, I'm [Your Name], and this is my machine learning project for Introduction to Machine Learning. Today I'll demonstrate a malware detection system built with LightGBM, deployed as a web application with full CI/CD automation.
>
> The project tackles binary classification on the Brazilian Malware Dataset—identifying whether executable files are goodware or malware. I trained and compared 7 different models, selected the best one based on cross-validation AUC, and deployed it to production."

**What to show:**
- Your face on camera briefly
- Screen showing the project title slide or GitHub repo

---

### **0:30-2:30 — Web Application Demo (2 minutes)**

**What to say:**
> "Let me show you the live web application deployed on Railway at this URL [read URL from screen]. The app has two main features: single instance prediction and batch file upload.
>
> First, I'll demonstrate single prediction. Here's the form with 26 input features—these are static properties extracted from executable files like PE header fields, section counts, and import statistics.
>
> Rather than entering all 26 manually, I'll click 'Load Malware Sample' to pre-fill the form with a real malware example from our test set."

**What to show:**
- Navigate to live URL: https://machine-learning-production-046b.up.railway.app
- Show the UI form with 26 input fields
- Click "Load Malware Sample" button
- Form fields populate

**What to say (continued):**
> "Now I'll click Predict. The model processes these features through our preprocessing pipeline—applying StandardScaler and handling categorical encoding—then returns a prediction. As you can see, it correctly identifies this as Malware with high confidence.
>
> Let me test a goodware sample. I'll click 'Load Goodware Sample' and predict again. The model correctly classifies this as Goodware."

**What to show:**
- Click "Predict" → Result shows "Malware (1)" with probability
- Click "Load Goodware Sample" → Click "Predict" → Result shows "Goodware (0)"

---

### **2:30-4:00 — Batch Upload & Metrics (1.5 minutes)**

**What to say:**
> "The second feature is batch prediction with CSV upload. I'll upload our 20% test set, which contains 4,243 samples with labels so we can evaluate model performance.
>
> When I click 'Upload and Predict,' the app processes all instances, generates predictions, and because this file includes the Label column, it automatically calculates evaluation metrics."

**What to show:**
- Scroll to "Batch Upload" section
- Click "Choose File" and select test CSV (show filename)
- Click "Upload and Predict"

**What to say (continued):**
> "Here are the results. The predictions table shows the first few rows with their classifications. Below that, we see the evaluation metrics:
> - **Test AUC: 0.8678** — strong discriminative power despite class imbalance
> - **Test Accuracy: 99.65%** — correctly classifies nearly all samples
> - **Confusion Matrix:** This is interesting—the model has perfect precision on malware (no false positives) but only 25% recall, meaning it missed 15 out of 20 malware samples. This is due to severe class imbalance—99.5% goodware in the dataset. The model is conservative, prioritizing low false alarms over catching all malware."

**What to show:**
- Predictions table displaying
- Metrics section with AUC, Accuracy
- Confusion Matrix visualization
- Point to specific numbers in confusion matrix

---

### **4:00-5:00 — Model Selection & Evaluation (1 minute)**

**What to say:**
> "Let me explain how I selected this model. I trained and compared 7 models using 10-fold stratified cross-validation.
>
> [Open evaluation-and-design.md or show CV results table]
>
> Here's the CV results table. The four baseline models—Logistic Regression, Decision Tree, Random Forest, and Neural Network—are at the bottom. The MLP struggled with class imbalance, achieving only random-guess AUC of 0.50.
>
> The three advanced models—XGBoost, LightGBM, and Gradient Boosting—all performed well. I selected **LightGBM** because it had the highest CV AUC of 0.9957 with the lowest variance of 0.0069, indicating stable performance across folds."

**What to show:**
- Open `Docs/evaluation-and-design.md` in browser or editor
- Scroll to CV results table
- Highlight LightGBM row

---

### **5:00-7:30 — CI/CD Pipeline & Testing (2.5 minutes)**

**What to say:**
> "Now let me demonstrate the CI/CD pipeline with automated testing. I'll make a small code change and push it to GitHub to show the full workflow.
>
> [Open GitHub repo in browser]
>
> This is my GitHub repository. Let me navigate to the Actions tab to show you the pipeline configuration. Here you can see previous workflow runs—tests execute on every push to main.
>
> [Open local code editor]
>
> I'll make a trivial change to trigger the pipeline—let me add a comment to the health endpoint in app.py."

**What to show:**
- GitHub repo main page
- Click "Actions" tab → Show previous workflow runs (green checkmarks)
- Click on one workflow run → Show test execution logs briefly
- Open VS Code → Open `Project/app/app.py`
- Add a comment like: `# Updated for demo - January 18, 2026`

**What to say (continued):**
> "Now I'll commit and push this change."

**What to show:**
- In terminal: 
  ```bash
  git add .
  git commit -m "Demo: trigger CI/CD pipeline"
  git push origin main
  ```

**What to say (continued):**
> "The push triggered the workflow. Let's watch it execute in real-time.
>
> [Go back to GitHub Actions tab and refresh]
>
> Here's the workflow running. It has two jobs: 'test' and 'deploy.' The test job is currently running—it installs dependencies, then executes our 47 unit and integration tests using pytest.
>
> [Wait for tests to pass, or if too slow, show a previous successful run]
>
> The tests passed! You can see all 47 tests executed successfully. This includes unit tests for the preprocessing pipeline, model wrapper, and API endpoints, plus integration tests for complete workflows.
>
> Now the deploy job starts—it only runs if tests pass. This job triggers Railway to rebuild and redeploy the application. In production, this takes about 2 minutes, but the important thing is that code only deploys if all tests pass, preventing broken code from reaching production."

**What to show:**
- GitHub Actions page with workflow running
- Show "test" job expanding → pytest output (if fast enough)
- Show "deploy" job starting after tests pass
- Point out the conditional logic: "needs: test"

---

### **7:30-8:30 — Code Reproducibility (1 minute)**

**What to say:**
> "All code is reproducible. Let me quickly show the repository structure.
>
> [Navigate GitHub repo file tree]
>
> We have the `data/` folder with train/test splits, the `models/` folder with the trained LightGBM model and preprocessor, `Project/app/` with the Flask application and tests, and the documentation files required for submission: evaluation-and-design.md, deployed.md, and ai-tooling.md.
>
> The preprocessing and training are reproducible with fixed random seeds. Anyone can clone this repo, run `pip install -r requirements.txt`, then execute `train.py` to reproduce the model training, or `eval.py` to reproduce the test set evaluation."

**What to show:**
- GitHub repo file explorer
- Quickly navigate to show:
  - `data/` folder
  - `models/` folder
  - `Project/app/` folder
  - `Docs/evaluation-and-design.md`
  - `Docs/deployed.md`
  - `ai-tooling.md`
  - `requirements.txt`

---

### **8:30-9:00 — Summary & Conclusion (30 seconds)**

**What to say:**
> "To summarize: I built a malware detection system with 7 models evaluated through cross-validation, selected LightGBM for production based on highest AUC, deployed it as a Flask web app with batch processing and metrics display, and implemented a full CI/CD pipeline with 47 automated tests that run before every deployment.
>
> The model achieves 99.65% accuracy and 0.87 AUC on the test set, with perfect precision on malware detection. All code is reproducible, documented, and live at the URL shown.
>
> Thank you for watching!"

**What to show:**
- Return to live web app URL briefly
- Or show your face on camera for closing

---

## SCORE 5 REQUIREMENTS CHECKLIST

Make sure your video demonstrates:

### ✅ Web Application Functionality
- [ ] Single instance prediction form
- [ ] Pre-filled demo data ("Load Sample" buttons)
- [ ] Prediction results displayed clearly
- [ ] Batch CSV upload feature
- [ ] Predictions table for batch results
- [ ] Evaluation metrics (AUC, accuracy, confusion matrix)

### ✅ Model Evaluation
- [ ] CV results table showing all 7 models
- [ ] Model comparison and selection reasoning
- [ ] Test set performance metrics
- [ ] Discussion of model strengths/limitations

### ✅ CI/CD Pipeline
- [ ] GitHub Actions workflow visible
- [ ] Tests running automatically on push
- [ ] Test results (47 tests passing)
- [ ] Auto-deploy triggered after tests pass
- [ ] Conditional deployment (only on test pass)

### ✅ Code Quality & Reproducibility
- [ ] GitHub repository structure shown
- [ ] Documentation files visible (evaluation-and-design.md, deployed.md, ai-tooling.md)
- [ ] Mention of reproducibility (requirements.txt, fixed seeds)
- [ ] train.py and eval.py scripts mentioned

### ✅ Presentation Quality
- [ ] You appear on camera (at least at start/end)
- [ ] You speak throughout the video
- [ ] Clear audio quality
- [ ] Professional delivery
- [ ] 5-10 minute duration
- [ ] All features demonstrated, not just described

---

## RECORDING SETUP

### Recommended Tools
**Screen Recording:**
- **OBS Studio** (Free, professional quality): https://obsproject.com
- **Zoom** (Record meeting with screen share + webcam)
- **Windows Game Bar** (Win+G, built-in)
- **macOS QuickTime** (Screen recording with audio)

**Camera Position:**
- Picture-in-picture in corner for full video
- OR full screen at intro/outro, screen share in middle

### Technical Settings
- **Resolution:** 1920x1080 (1080p) or 1280x720 (720p minimum)
- **Frame Rate:** 30fps minimum
- **Audio:** Clear microphone, no background noise
- **File Format:** MP4 (H.264 codec)
- **File Size:** Keep under 500MB if uploading directly

### Pre-Recording Checklist
- [ ] Close unnecessary browser tabs and apps
- [ ] Turn off notifications (Do Not Disturb mode)
- [ ] Have all URLs bookmarked and ready
- [ ] Test CSV file ready for upload demo
- [ ] GitHub repo page ready in browser
- [ ] VS Code open with app.py ready
- [ ] Terminal ready with git commands
- [ ] Camera and microphone tested
- [ ] Clean desktop/background

---

## PRACTICE PLAN

### Run-Through 1: Dry Run
- Read script without recording
- Navigate all URLs and apps
- Time yourself (aim for 7-9 minutes)
- Identify any technical issues

### Run-Through 2: Recorded Practice
- Record but don't save
- Watch yourself back
- Note areas to improve:
  - Speaking pace (too fast/slow?)
  - Screen navigation (smooth transitions?)
  - Audio quality (clear?)
  - Timing (each section on track?)

### Run-Through 3: Final Recording
- Set up environment (close apps, mute notifications)
- Record full video
- Review immediately after
- Re-record if major issues
- Save best take

---

## COMMON MISTAKES TO AVOID

❌ **Don't:**
- Rush through features without showing them
- Just describe what the app does—DEMONSTRATE it
- Forget to show yourself on camera
- Have audio issues (test first!)
- Go over 10 minutes or under 5 minutes
- Show errors or broken features
- Read directly from script (speak naturally)
- Show sensitive information (API keys, passwords)

✅ **Do:**
- Show every feature in action
- Explain WHILE demonstrating
- Speak clearly and confidently
- Make eye contact with camera
- Show enthusiasm about your work
- Point to specific elements on screen
- Test everything before recording
- Have clean, professional presentation

---

## POST-RECORDING

### Video Review Checklist
- [ ] Duration is 5-10 minutes
- [ ] You appear on camera and speak throughout
- [ ] All features demonstrated (not just mentioned)
- [ ] Audio is clear throughout
- [ ] No long pauses or dead air
- [ ] No technical errors shown
- [ ] Video quality is HD
- [ ] Professional presentation

### Upload Instructions
1. **YouTube (Recommended):**
   - Upload as "Unlisted" (not Public, not Private)
   - Title: "ML Project - Brazilian Malware Detection"
   - Copy unlisted link for submission

2. **Google Drive:**
   - Upload MP4 file
   - Set sharing to "Anyone with the link can view"
   - Copy shareable link

3. **Vimeo:**
   - Upload with password protection optional
   - Copy shareable link

---

## FINAL SUBMISSION

### Create Submission PDF
Include:
1. **Link 1:** Demo video URL (YouTube/Drive/Vimeo)
2. **Link 2:** GitHub repository URL
3. Brief intro (1-2 sentences explaining the links)

**Example PDF Content:**
```
Machine Learning Project Submission
Introduction to Machine Learning

Student: [Your Name]

Demo Video: https://youtu.be/[your-video-id]
GitHub Repository: https://github.com/[username]/machine-learning

This project demonstrates a complete machine learning pipeline for malware 
detection, including model training, web deployment, and CI/CD automation.
```

### Submission Steps
1. Click "Submit Project" on dashboard
2. Upload PDF with links
3. Confirm submission
4. Done! ✅

---

## ESTIMATED TIMELINE

| Task | Time Needed |
|------|-------------|
| Practice run-through | 15 minutes |
| Setup recording environment | 10 minutes |
| Record (including retakes) | 45-60 minutes |
| Review and trim if needed | 15 minutes |
| Upload to YouTube/Drive | 10 minutes |
| Create submission PDF | 5 minutes |
| Submit on dashboard | 5 minutes |
| **Total** | **~2 hours** |

---

**You're fully prepared for a Score 5 submission! Good luck with your recording!** 🎥✨

**Live URL:** https://machine-learning-production-046b.up.railway.app
**GitHub:** [Your repo URL here]
