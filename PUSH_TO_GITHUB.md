# 🚀 Complete Commands to Push to GitHub

## ✅ Git LFS is Already Set Up!

Git LFS is installed and configured. Now follow these steps:

## 📋 Step-by-Step Commands

### 1. Add the Model File (with Git LFS)
```bash
git add model_repo/weights/swin_nih_chestxray14.pth
```

### 2. Add All Other Files
```bash
git add .
```

### 3. Commit Everything
```bash
git commit -m "Add VisioLung.ai project with pretrained model (swin_nih_chestxray14.pth)"
```

### 4. Check if Remote Exists
```bash
git remote -v
```

### 5A. If Remote Doesn't Exist - Add It
```bash
# Replace with your GitHub username and repository name
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

### 5B. If Remote Exists - Update It (if needed)
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

### 6. Push to GitHub
```bash
git push -u origin main
```

## ⚠️ Important Notes

1. **First Push Time:** The first push with the model file may take 10-30 minutes depending on:
   - File size (~500MB)
   - Your internet speed
   - GitHub's servers

2. **Verify LFS Tracking:**
   ```bash
   git lfs ls-files
   ```
   This should show your `.pth` file is tracked by Git LFS

3. **GitHub LFS Limits:**
   - Free tier: 1GB storage, 1GB bandwidth/month
   - Your model should fit within free tier

## 🔍 Quick Verification Commands

After pushing, verify:
```bash
# Check LFS files
git lfs ls-files

# Check remote
git remote -v

# Check status
git status
```

## 📝 Complete Command Sequence (Copy & Paste)

```bash
# Add model file
git add model_repo/weights/swin_nih_chestxray14.pth

# Add all other files
git add .

# Commit
git commit -m "Add VisioLung.ai project with pretrained model"

# Add remote (replace YOUR_USERNAME and YOUR_REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push
git push -u origin main
```

## 🆘 Troubleshooting

### If push fails:
1. Make sure you have a GitHub repository created
2. Check your internet connection
3. Verify Git LFS is working: `git lfs ls-files`
4. Try pushing again: `git push -u origin main`

### If you need to create a new GitHub repo:
1. Go to https://github.com/new
2. Name your repository (e.g., "VisioLung.ai")
3. Don't initialize with README
4. Copy the repository URL
5. Use it in step 5A above

