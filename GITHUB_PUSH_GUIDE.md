# 🚀 Push Project to GitHub with Large Model File

## Step-by-Step Commands

### Step 1: Install Git LFS (if not already installed)

**Windows:**
```bash
# Download and install from: https://git-lfs.github.com/
# Or use Chocolatey:
choco install git-lfs
```

**Mac:**
```bash
brew install git-lfs
```

**Linux:**
```bash
sudo apt-get install git-lfs
```

### Step 2: Initialize Git LFS

```bash
git lfs install
```

### Step 3: Track Large Files

```bash
git lfs track "*.pth"
git lfs track "model_repo/weights/*.pth"
```

### Step 4: Add .gitattributes

```bash
git add .gitattributes
```

### Step 5: Add All Files

```bash
# Add the model file first (with LFS)
git add model_repo/weights/swin_nih_chestxray14.pth

# Add all other files
git add .

# Or add specific directories
git add backend/
git add frontend/
git add model_repo/
```

### Step 6: Commit

```bash
git commit -m "Initial commit: Add VisioLung.ai project with pretrained model"
```

### Step 7: Create GitHub Repository (if not exists)

1. Go to https://github.com/new
2. Create a new repository (don't initialize with README)
3. Copy the repository URL

### Step 8: Add Remote and Push

```bash
# Add remote (replace with your GitHub URL)
git remote add origin https://github.com/your-username/VisioLung.ai.git

# Or if remote already exists, update it:
git remote set-url origin https://github.com/your-username/VisioLung.ai.git

# Push to GitHub
git push -u origin main
```

## ⚠️ Important Notes

1. **Git LFS Limits:**
   - GitHub free tier: 1GB storage, 1GB bandwidth/month
   - Your model file (~500MB) should fit within free tier
   - If you exceed limits, consider:
     - Using GitHub LFS paid plans
     - Storing model on cloud storage (S3, GCS) and downloading during deployment

2. **First Push:**
   - First push with LFS files may take 10-30 minutes depending on file size and internet speed
   - Be patient and don't interrupt the process

3. **Verify LFS:**
   ```bash
   git lfs ls-files
   ```
   This should show your .pth file tracked by LFS

## 🔍 Troubleshooting

### If Git LFS is not installed:
```bash
# Windows: Download from https://git-lfs.github.com/
# Then run:
git lfs install
```

### If push fails due to size:
- Make sure Git LFS is properly installed and initialized
- Verify .gitattributes is committed
- Check file is tracked: `git lfs ls-files`

### If you need to update remote URL:
```bash
git remote -v  # Check current remote
git remote set-url origin https://github.com/your-username/your-repo.git
```

## ✅ Quick Command Sequence

```bash
# 1. Install Git LFS (if needed)
# Download from: https://git-lfs.github.com/

# 2. Initialize and setup
git lfs install
git lfs track "*.pth"
git add .gitattributes

# 3. Add files
git add model_repo/weights/swin_nih_chestxray14.pth
git add .

# 4. Commit
git commit -m "Add VisioLung.ai project with pretrained model"

# 5. Push (replace with your GitHub URL)
git remote add origin https://github.com/your-username/VisioLung.ai.git
git push -u origin main
```

## 📝 Alternative: If Git LFS is Not Available

If you can't use Git LFS, you have these options:

1. **Exclude model from Git:**
   - Add to `.gitignore`: `model_repo/weights/*.pth`
   - Upload model separately to cloud storage
   - Download during deployment

2. **Use Git LFS alternatives:**
   - GitHub Releases (attach model as release asset)
   - Cloud storage (S3, Google Drive, etc.)
   - Model hosting services (Hugging Face, etc.)

