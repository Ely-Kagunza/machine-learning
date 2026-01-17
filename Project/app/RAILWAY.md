# Railway Deployment Guide

## Step-by-Step Setup

### 1. Create Railway Account
- Go to https://railway.app
- Click **"Start a New Project"**
- Sign in with GitHub (recommended)

### 2. Deploy Your Project

**Option A: Deploy from GitHub (Recommended)**
1. Click **"Deploy from GitHub repo"**
2. Select your repository: `Ely-Kagunza/machine-learning`
3. Railway will detect it's a Python app automatically

**Option B: Deploy from CLI**
```bash
npm i -g @railway/cli
railway login
railway init
railway up
```

### 3. Configure Settings

After connecting your repo, configure these settings:

#### **Root Directory**
- Go to **Settings** → **Build**
- Set **Root Directory**: `Project/app`
- Save changes

#### **Environment Variables**
- Go to **Variables** tab
- Add: `PYTHON_VERSION` = `3.11.9` (optional, auto-detected)
- Railway auto-sets `PORT` variable

#### **Start Command** (Auto-detected from Procfile)
Railway will use: `gunicorn --config gunicorn.conf.py app:app`

### 4. Deploy
- Railway auto-deploys on every GitHub push
- First deploy takes 5-10 minutes
- You'll get a URL like: `https://your-app.up.railway.app`

### 5. Monitor Deployment
- **Build Logs**: Check for errors during pip install
- **Deploy Logs**: Check for runtime errors
- **Metrics**: View memory/CPU usage

## Railway vs Render

| Feature | Railway | Render |
|---------|---------|--------|
| Free Credit | $5/month | 750 hours/month |
| Memory Handling | Better | Stricter |
| Cold Starts | Faster | Slower |
| Build Speed | Faster | Slower |
| Setup | Easier | More config |

## Troubleshooting

### Build Fails
- Check **Build Logs** in Railway dashboard
- Verify `requirements.txt` has all dependencies
- Ensure Root Directory is `Project/app`

### App Crashes
- Check **Deploy Logs**
- Verify model files exist at `../../models/`
- Check memory usage in Metrics tab

### Out of Memory
- Railway free tier has 512MB RAM (same as Render)
- But better memory management
- If still issues, upgrade to Hobby plan ($5/month for 8GB)

## Updating Your App

Push to GitHub and Railway auto-deploys:
```bash
git add .
git commit -m "Update app"
git push origin main
```

Deployment takes 2-5 minutes.

## Custom Domain (Optional)

1. Go to **Settings** → **Networking**
2. Click **Generate Domain** for free Railway subdomain
3. Or add custom domain (requires DNS setup)

## Cost Management

- Free tier: $5 credit/month (enough for ~500 hours)
- Monitor usage in **Billing** tab
- App sleeps after inactivity (same as Render)
- Upgrade to Hobby ($5/month) for no sleep + more resources

## Support

- Railway Docs: https://docs.railway.app
- Discord: https://discord.gg/railway
- Status: https://status.railway.app
