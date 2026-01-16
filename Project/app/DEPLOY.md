# Deployment Guide - Render

## Prerequisites
- GitHub account
- Render account (free): https://render.com

## Quick Deployment Steps

### 1. Push to GitHub
```bash
# Initialize git if not already done
git init
git add .
git commit -m "Ready for deployment"

# Create a new repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/malware-detection.git
git branch -M main
git push -u origin main
```

### 2. Deploy on Render

1. Go to https://render.com and sign in
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Render will auto-detect `render.yaml` and configure everything

**Manual Configuration (if needed):**
- **Name**: malware-detection
- **Region**: Oregon (US West)
- **Branch**: main
- **Root Directory**: `Project/app`
- **Environment**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app`
- **Plan**: Free

5. Click **"Create Web Service"**

### 3. Wait for Deployment
- First build takes 5-10 minutes
- You'll get a URL like: `https://malware-detection-xxxx.onrender.com`

### 4. Test Your Deployment
Visit your URL and test:
- Single prediction form
- Load sample data buttons
- Batch CSV upload with the sample file

## Important Notes

### Free Tier Limitations
- **Cold starts**: App sleeps after 15 min of inactivity (~30s to wake up)
- **750 hours/month**: More than enough for personal/demo use
- **Build time**: ~5-10 min on first deploy
- **No auto-scaling**: 512MB RAM (sufficient for your model)

### File Structure for Deployment
```
Project/app/
├── app.py                 # Main Flask app
├── requirements.txt       # Python dependencies
├── render.yaml           # Render configuration
├── Procfile              # Alternative start command
├── static/               # CSS, JS
├── templates/            # HTML
└── .env.example          # Environment template
```

### Models Directory
Make sure your models are committed:
```bash
# Check if models are in .gitignore
cat ../../../.gitignore

# If models are ignored, you need to:
# Option 1: Remove models from .gitignore and commit them
# Option 2: Use Render Disk storage (paid feature)
# Option 3: Download models on startup (add to buildCommand)
```

### Troubleshooting

**Build fails:**
- Check build logs in Render dashboard
- Verify all paths in `app.py` are relative
- Ensure models directory is accessible

**App crashes:**
- Check runtime logs
- Verify model files exist in correct path
- Check Python version compatibility

**Slow cold starts:**
- Normal for free tier
- Consider upgrading to paid tier ($7/month) for 0 cold starts

## Alternative: Deploy from Render Dashboard

If auto-detection fails:

1. **Environment Variables** (none required for basic deployment)
2. **Advanced Settings**:
   - Set Root Directory: `Project/app`
   - Health Check Path: `/health`

## Updating Your App

After making changes:
```bash
git add .
git commit -m "Update feature"
git push origin main
```

Render will automatically rebuild and redeploy (takes 2-5 min).

## Cost
- **Free tier**: $0/month (sufficient for testing & demos)
- **Starter tier**: $7/month (no cold starts, better performance)
- **Standard tier**: $25/month (2GB RAM, autoscaling)

## Support
- Render docs: https://render.com/docs
- Community: https://community.render.com
