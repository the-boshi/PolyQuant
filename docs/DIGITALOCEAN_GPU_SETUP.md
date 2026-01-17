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

Once the droplet is running, get the IP address from the console.

### Option A: Terminal SSH

```bash
ssh root@<YOUR_DROPLET_IP>
```

### Option B: VS Code Remote SSH (Recommended)

Connect VS Code directly to the droplet for a full IDE experience:

1. **Install the extension** in VS Code:
   - Open Extensions (`Ctrl+Shift+X`)
   - Search for **"Remote - SSH"** by Microsoft
   - Install it

2. **Connect to the droplet**:
   - Press `F1` → type **"Remote-SSH: Connect to Host"**
   - Enter: `root@<YOUR_DROPLET_IP>`
   - Select **Linux** when prompted
   - Enter your SSH key passphrase

3. **Open the project folder**:
   - Once connected, click **"Open Folder"**
   - Navigate to `/root/polyquant`

**Benefits of VS Code Remote SSH:**
- Full VS Code experience on the remote machine
- File explorer with all remote files
- Integrated terminal (`Ctrl+``) opens SSH terminal directly
- Extensions (Python, Pylance) run remotely
- Git integration works with the remote repo
- Edit code and run training in the same window

**Tips:**
- Install the Python extension on the remote when prompted
- Run `watch -n 1 nvidia-smi` in a terminal to monitor GPU
- Use `tmux` inside the VS Code terminal for persistent sessions

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

### Which Folders to Upload?

| Folder | Upload? | Used By |
|--------|---------|---------|
| `features_full/` | ✅ **YES** | ResNet training |
| `sequences/` | ✅ **YES** | Dual encoder (market sequences) |
| `user_sequences_store/` | ✅ **YES** | Dual encoder (user sequences) |
| `features/` | ❌ No | Raw data - can regenerate features_full |
| `features_dataset_downsampled/` | ❌ No | Old version, not used |
| `user_sequences/` | ❌ No | Intermediate data for building store |
| `market_meta.parquet` | ❌ No | Already embedded in features_full |

### On Windows (PowerShell):

```powershell
# Navigate to data directory
cd E:\Roy_Data\Projects\Technion\deep-project\PolyQuant\data

# Compress folders separately (easier to manage)
tar -cvzf features_full.tar.gz features_full
tar -cvzf sequences.tar.gz sequences
tar -cvzf user_sequences_store.tar.gz user_sequences_store

# Check sizes before upload
Get-ChildItem *.tar.gz | Select-Object Name, @{Name="SizeGB";Expression={[math]::Round($_.Length/1GB, 2)}}

# Transfer all to droplet
scp features_full.tar.gz sequences.tar.gz user_sequences_store.tar.gz root@<YOUR_DROPLET_IP>:/root/polyquant/data/
```

### On the Droplet:

```bash
cd /root/polyquant/data

# Extract all archives
tar -xvzf features_full.tar.gz
tar -xvzf sequences.tar.gz
tar -xvzf user_sequences_store.tar.gz

# Remove archives to save space
rm *.tar.gz

# Verify the data
ls -la features_full/train/ | head
ls -la sequences/
ls -la user_sequences_store/
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

GPU droplets are billed **per hour from the moment they are created**, even when idle or powered off. The only way to stop billing is to **destroy the droplet**.

### Pricing Overview

| Duration | Cost (H200 @ $3.44/hr) |
|----------|------------------------|
| 1 hour | $3.44 |
| 8 hours | $27.52 |
| 24 hours | $82.56 |
| 1 week | $578 |

### Using Snapshots to Save Money

Snapshots let you save your configured droplet state, destroy the droplet to stop billing, and restore it later.

**Snapshot Pricing**: $0.06 per GB per month

For example, if your snapshot is 100GB:
- Monthly storage cost: $6/month ($0.20/day)
- Much cheaper than keeping a $3.44/hr droplet running

### Creating a Snapshot

1. **Via DigitalOcean Console:**
   - Go to your droplet's page
   - Click **Snapshots** in the left sidebar
   - Click **Take Snapshot**
   - Enter a name (e.g., `polyquant-configured-2026-01-17`)
   - Wait for completion (can take 10-30 minutes depending on size)

2. **Via CLI (doctl):**
   ```bash
   # Install doctl if needed
   # See: https://docs.digitalocean.com/reference/doctl/how-to/install/

   # Authenticate
   doctl auth init

   # List your droplets to get the ID
   doctl compute droplet list

   # Create snapshot (replace DROPLET_ID)
   doctl compute droplet-action snapshot DROPLET_ID --snapshot-name "polyquant-configured"
   ```

### Recommended Workflow

1. **First time setup:**
   - Create droplet
   - Set up Python environment, install dependencies
   - Transfer data files
   - **Create a snapshot** (before running any training)
   - This snapshot is your "ready to train" baseline

2. **Training session:**
   - Restore from snapshot (or create new droplet from snapshot)
   - Pull latest code: `git pull`
   - Run training
   - Download checkpoints to local machine
   - **Destroy the droplet** (not just power off!)

3. **Next training session:**
   - Create new droplet from your snapshot
   - Your environment and data are already there
   - Just `git pull` and start training

### Restoring from a Snapshot

1. Go to **Images** → **Snapshots** in DigitalOcean console
2. Find your snapshot
3. Click **More** → **Create Droplet**
4. Select the same GPU droplet type
5. Your droplet will boot with everything intact

### Important Notes

- **Power Off ≠ Stop Billing**: A powered-off droplet still costs money
- **Only Destroy Stops Billing**: You must destroy the droplet to stop charges
- **Snapshots persist**: Your snapshot remains even after destroying the droplet
- **Data not in snapshot**: If you generated new data/checkpoints, download them before destroying!

### Cost Comparison Example

**Scenario**: You train for 8 hours, then wait 1 week before next session

| Approach | Cost |
|----------|------|
| Keep droplet running (24/7 for 1 week) | $578 |
| Destroy + Snapshot (100GB, 1 week) | $27.52 (training) + $1.40 (storage) = **$28.92** |

**Savings: ~$549**

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
