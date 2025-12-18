# Complete RTX 3090 Setup Guide: Georgian Speech-to-Text

This guide assumes **nothing is set up** on your RTX 3090 machine. Follow each step in order.

---

## PHASE 1: System Setup (One-Time)

### Step 1.1: Install NVIDIA Drivers (if not installed)

```bash
# Check if NVIDIA drivers are installed
nvidia-smi

# If not installed, on Ubuntu:
sudo apt update
sudo apt install nvidia-driver-535  # or latest version
sudo reboot
```

### Step 1.2: Install CUDA Toolkit

```bash
# Check CUDA version
nvcc --version

# If not installed, download from NVIDIA:
# https://developer.nvidia.com/cuda-downloads
# For Ubuntu 22.04:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-4
```

### Step 1.3: Install Python 3.10+ and pip

```bash
# Check Python version
python3 --version

# If not installed or outdated:
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

### Step 1.4: Install system audio libraries

```bash
sudo apt install ffmpeg libsndfile1
```

---

## PHASE 2: Project Setup

### Step 2.1: Transfer files to RTX 3090 machine

**Option A: Using the package script (from Mac/source machine)**
```bash
# On the source machine:
cd /Users/lana/Desktop/speech_to_text
./package_for_training.sh

# This creates package_transfer/ folder (~7GB)
# Transfer via scp, rsync, or USB drive:
scp -r package_transfer/ user@rtx3090-machine:~/speech_to_text/
```

**Option B: Direct copy (if accessible)**
```bash
# Copy entire directory
scp -r /Users/lana/Desktop/speech_to_text user@rtx3090-machine:~/speech_to_text/
```

### Step 2.2: Create Python virtual environment

```bash
cd ~/speech_to_text
python3 -m venv venv
source venv/bin/activate
```

### Step 2.3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 2.4: Install PyTorch with CUDA support

```bash
# IMPORTANT: Install PyTorch FIRST with CUDA support
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2.5: Install remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 2.6: Verify GPU is detected

```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
```

---

## PHASE 3: Prepare Dataset

### Step 3.1: Extract dataset (if packaged)

```bash
cd ~/speech_to_text

# If you used package_for_training.sh:
tar -xzf cv-corpus-ka-dataset.tar.gz
```

### Step 3.2: Verify dataset structure

```bash
ls -la cv-corpus-23.0-2025-09-05/ka/
```

Expected files:
- `clips/` folder (contains audio files)
- `train.tsv`
- `test.tsv`
- `dev.tsv`

### Step 3.3: Quick dataset check

```bash
# Count audio files
ls cv-corpus-23.0-2025-09-05/ka/clips/ | wc -l
# Expected: ~227,000 files

# Check TSV files exist
head -2 cv-corpus-23.0-2025-09-05/ka/train.tsv
```

---

## PHASE 4: Training

### Step 4.1: (Optional) Run baseline test first

```bash
source venv/bin/activate
python 0_quick_test.py
```

This shows how the pre-trained model performs on Georgian BEFORE fine-tuning.

### Step 4.2: Start training

```bash
source venv/bin/activate
python 2_finetune_whisper_local.py
```

**What happens:**
- Script auto-detects RTX 3090 (24GB VRAM)
- Sets optimal batch size (16) and gradient accumulation (1)
- Downloads whisper-small model (~1.5GB)
- Trains for 3 epochs on Georgian data
- **Duration: 8-12 hours**

### Step 4.3: Monitor training progress

**Option A: Watch console output**
- Shows loss values decreasing
- Shows progress bar with ETA

**Option B: Use TensorBoard (recommended)**
```bash
# In a new terminal:
source venv/bin/activate
tensorboard --logdir ./whisper-georgian-local
# Open browser: http://localhost:6006
```

### Step 4.4: Handle interruptions

If training is interrupted (Ctrl+C or power loss):
```bash
# The script auto-saves checkpoints
# Just restart - it will resume from last checkpoint:
python 2_finetune_whisper_local.py
```

### Step 4.5: Training complete

When training finishes, you'll see:
```
Training completed!
Model saved to: ./whisper-georgian-local
```

Verify the model exists:
```bash
ls -la whisper-georgian-local/
```

Expected files:
- `config.json`
- `model.safetensors` (or `pytorch_model.bin`)
- `preprocessor_config.json`
- `tokenizer.json`
- etc.

---

## PHASE 5: Test the Trained Model

### Step 5.1: Quick transcription test

```bash
source venv/bin/activate

# Test with a sample from the dataset
python 3_transcribe.py --audio cv-corpus-23.0-2025-09-05/ka/clips/common_voice_ka_17186378.mp3 --model ./whisper-georgian-local
```

### Step 5.2: Transcribe your own audio file

```bash
python 5_transcribe_to_file.py --audio /path/to/your/audio.mp3 --model ./whisper-georgian-local
```

Output: Creates `your_audio_transcription.txt` with the transcription.

### Step 5.3: Batch transcribe multiple files

```bash
# Create a folder with audio files
mkdir my_audio_files
# Copy your audio files to my_audio_files/

# Transcribe all of them
python 5_transcribe_to_file.py --audio-folder ./my_audio_files/ --model ./whisper-georgian-local --output-folder ./transcriptions/
```

### Step 5.4: Launch web interface

```bash
# Local only (access from same machine)
python 6_web_interface.py --model ./whisper-georgian-local

# Or with network access (access from other machines)
python 6_web_interface.py --model ./whisper-georgian-local --server-name 0.0.0.0

# Or with public shareable link
python 6_web_interface.py --model ./whisper-georgian-local --share
```

Open browser: http://localhost:7860
- Upload audio file or record from microphone
- Click "Transcribe"
- Download the .txt file with transcription

---

## QUICK REFERENCE: Command Order Summary

```bash
# === ONE-TIME SETUP ===
# 1. Install NVIDIA drivers (if needed)
sudo apt install nvidia-driver-535

# 2. Install system dependencies
sudo apt install python3.10 python3.10-venv python3-pip ffmpeg libsndfile1

# 3. Create virtual environment
cd ~/speech_to_text
python3 -m venv venv
source venv/bin/activate

# 4. Install PyTorch with CUDA
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install other dependencies
pip install -r requirements.txt

# 6. Verify GPU
python3 -c "import torch; print(torch.cuda.is_available())"

# === TRAINING (8-12 hours) ===
source venv/bin/activate
python 2_finetune_whisper_local.py

# === TESTING ===
# Option 1: Command line
python 5_transcribe_to_file.py --audio your_file.mp3 --model ./whisper-georgian-local

# Option 2: Web interface
python 6_web_interface.py --model ./whisper-georgian-local
# Open: http://localhost:7860
```

---

## Troubleshooting

### GPU not detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA if needed
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Out of memory during training
```bash
# Edit 2_finetune_whisper_local.py and reduce batch size
# Or set environment variable:
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
python 2_finetune_whisper_local.py
```

### Model not found error
```bash
# Make sure you're in the right directory
cd ~/speech_to_text

# Check model exists
ls whisper-georgian-local/
```

### Audio file not supported
```bash
# Install ffmpeg
sudo apt install ffmpeg

# Convert to supported format
ffmpeg -i input.m4a -ar 16000 output.wav
```

---

## Expected Results

| Metric | Before Fine-tuning | After Fine-tuning |
|--------|-------------------|-------------------|
| Word Error Rate (WER) | 40-60% | 10-20% |
| Output Quality | Often gibberish | Accurate Georgian |
| Improvement | - | 3-4x better |

---

## Files Created by Training

```
whisper-georgian-local/
├── config.json                 # Model configuration
├── model.safetensors          # Model weights (~1.7GB)
├── preprocessor_config.json   # Audio preprocessing config
├── tokenizer.json             # Tokenizer
├── vocab.json                 # Vocabulary
├── special_tokens_map.json    # Special tokens
├── generation_config.json     # Generation settings
└── training_args.bin          # Training arguments
```

---

## Timeline Summary

| Phase | Duration | Description |
|-------|----------|-------------|
| System Setup | 30-60 min | Drivers, CUDA, Python |
| Project Setup | 15-30 min | Virtual env, dependencies |
| Data Prep | 5-10 min | Extract and verify dataset |
| Training | 8-12 hours | Fine-tune whisper-small |
| Testing | 5 min | Verify model works |
| **Total** | **~9-14 hours** | Mostly training time |

---

Created: December 2024
Hardware: NVIDIA RTX 3090 (24GB VRAM)
Model: whisper-small (244M parameters)
Dataset: Mozilla Common Voice Georgian v23.0
