# DigitalOcean GPU Droplet Setup Guide

This guide walks you through setting up the PolyQuant project on a DigitalOcean GPU droplet with NVIDIA H200.

## Prerequisites

- DigitalOcean account with GPU droplet access
- GitHub repository with your code
- SSH key pair on your local machine
- Data prepared in `data/features_full/` directory

---

## Step 1: Create the GPU Droplet

1. Go to [DigitalOcean Cloud Console](https://cloud.digitalocean.com/)
2. Click **Create** → **Droplets**
3. Choose **GPU Droplets** tab
4. Select the **H200** instance type
5. Choose a region close to you
6. For the OS image, select **Ubuntu 22.04** with **NVIDIA GPU drivers pre-installed** (or the ML-ready image)
7. Choose **SSH Key** authentication and add your public key
8. Click **Create Droplet**

---

## Step 2: Connect to Your Droplet

Once the droplet is running, get the IP address from the console:

```bash
ssh root@<YOUR_DROPLET_IP>
```

---

## Step 3: Initial Server Setup

Run these commands on the droplet:

```bash
# Update system
apt update && apt upgrade -y

# Verify GPU is detected
nvidia-smi

# Install Python and essentials
apt install -y python3.11 python3.11-venv python3-pip git tmux htop

# Create project directory
mkdir -p /root/polyquant
```

---

## Step 4: Set Up GitHub SSH Access (for private repos)

If your repository is private, set up SSH authentication:

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Display the public key
cat ~/.ssh/id_ed25519.pub
```

Copy the output and add it to GitHub:
- Go to **GitHub → Settings → SSH and GPG keys → New SSH key**
- Paste the key and save

Test the connection:
```bash
ssh -T git@github.com
```

---

## Step 5: Clone the Repository

```bash
cd /root

# For private repo (SSH):
git clone git@github.com:<YOUR_USERNAME>/PolyQuant.git polyquant

# For public repo (HTTPS):
# git clone https://github.com/<YOUR_USERNAME>/PolyQuant.git polyquant

cd polyquant
```

---

## Step 6: Set Up Python Environment

```bash
cd /root/polyquant

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project requirements
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

---

## Step 7: Transfer Data Files

Data files are not stored in git. Transfer them from your local machine.

### On Windows (PowerShell):

```powershell
# Navigate to data directory
cd E:\Roy_Data\Projects\Technion\deep-project\PolyQuant\data

# Compress the features_full directory
tar -cvzf features_full.tar.gz features_full

# Transfer to droplet
scp features_full.tar.gz root@<YOUR_DROPLET_IP>:/root/polyquant/data/

# Transfer market_meta if needed
scp market_meta.parquet root@<YOUR_DROPLET_IP>:/root/polyquant/data/
```

### On the Droplet:

```bash
cd /root/polyquant/data

# Extract the data
tar -xvzf features_full.tar.gz

# Remove the archive to save space
rm features_full.tar.gz

# Verify the data
ls -la features_full/train/ | head
```

---

## Step 8: Configure the Project

Update `config.json` for the Linux environment:

```bash
cd /root/polyquant

cat > config.json << 'EOF'
{
  "root": ".",
  "dataset_root": "data/features_full",
  "scaler_path": "data/features_full/train_scaler.json",
  "runs_dir": "runs",
  "checkpoints_dir": "checkpoints"
}
EOF
```

---

## Step 9: Run Training

Use `tmux` to keep the training running even if you disconnect:

```bash
# Start a new tmux session
tmux new -s train

# Activate environment
cd /root/polyquant
source venv/bin/activate

# Start training
python polyquant/training/train_resnet.py
```

### tmux Commands:
- **Detach from session**: `Ctrl+B`, then `D`
- **Reattach to session**: `tmux attach -t train`
- **List sessions**: `tmux ls`
- **Kill session**: `tmux kill-session -t train`

---

## Useful Commands

### Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Check Disk Space
```bash
df -h
```

### View Training Logs
```bash
tail -f runs/<run_name>/train.log
```

### Pull Latest Code Changes
```bash
cd /root/polyquant
git pull
```

---

## Syncing Code Changes

### From Local Machine (after making changes):
```powershell
git add .
git commit -m "description of changes"
git push
```

### On the Droplet:
```bash
cd /root/polyquant
git pull
```

---

## Downloading Checkpoints

To download trained checkpoints back to your local machine:

```powershell
# From Windows PowerShell
scp -r root@<YOUR_DROPLET_IP>:/root/polyquant/checkpoints/<run_name> E:\Roy_Data\Projects\Technion\deep-project\PolyQuant\checkpoints\
```

---

## Cost Management

- **Stop the droplet** when not in use to avoid charges
- GPU droplets are billed per hour, even when idle
- Consider using **Snapshots** to save the configured state before destroying

---

## Troubleshooting

### GPU not detected
```bash
# Check NVIDIA drivers
nvidia-smi

# If not working, install drivers
apt install -y nvidia-driver-535
reboot
```

### CUDA version mismatch
```bash
# Check CUDA version
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu<VERSION>
```

### Out of disk space
```bash
# Check usage
du -sh /root/polyquant/*

# Clean up old runs/checkpoints if needed
rm -rf runs/old_run_name
rm -rf checkpoints/old_checkpoint
```
