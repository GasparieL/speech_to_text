# Local Training Guide for Whisper Fine-tuning

This guide explains how to train the Whisper model locally on your own machine.

## Table of Contents
1. [Hardware Requirements](#hardware-requirements)
2. [Software Setup](#software-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Configuration](#configuration)
5. [Running Training](#running-training)
6. [Monitoring Progress](#monitoring-progress)
7. [Troubleshooting](#troubleshooting)

---

## Hardware Requirements

### Recommended Setup

| Hardware | Minimum | Recommended | Optimal |
|----------|---------|-------------|---------|
| **GPU** | None (CPU works but slow) | 8GB VRAM (RTX 3060) | 24GB VRAM (RTX 3090/4090) |
| **RAM** | 8GB | 16GB | 32GB+ |
| **Storage** | 20GB free | 50GB free | 100GB+ free |
| **CPU** | 4 cores | 8 cores | 16+ cores |

### Model Size vs Hardware

| Model | Parameters | VRAM Needed | Quality | Speed |
|-------|-----------|-------------|---------|-------|
| whisper-tiny | 39M | ~1GB | Basic | Very Fast |
| whisper-base | 74M | ~1GB | Good | Fast |
| whisper-small | 244M | ~2GB | Better | Medium |
| whisper-medium | 769M | ~5GB | Great | Slow |
| whisper-large-v3 | 1550M | ~10GB | Best | Very Slow |

**Recommendation**: Start with `whisper-small` - it's a good balance of quality and speed.

---

## Software Setup

### Step 1: Install Python and Dependencies

```bash
# Check Python version (need 3.8+)
python --version

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For NVIDIA GPU
# OR for CPU only:
# pip install torch torchvision torchaudio

# Install other dependencies
pip install transformers datasets evaluate jiwer pandas tensorboard
```

### Step 2: Verify GPU Setup (if using GPU)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output if GPU is working:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
```

---

## Dataset Preparation

### Option 1: Common Voice Dataset (Recommended)

1. **Download Dataset**:
   - Go to [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets)
   - Select Georgian language
   - Download the dataset (requires account)
   - Extract to your project directory

2. **Verify Structure**:
   ```
   cv-corpus-23.0-2025-09-05/ka/
   ├── clips/              # Audio files
   ├── train.tsv          # Training metadata
   ├── test.tsv           # Test metadata
   └── dev.tsv            # Validation metadata
   ```

### Option 2: Your Own Dataset

Create this structure:
```
your_dataset/
├── clips/
│   ├── audio001.mp3
│   ├── audio002.mp3
│   └── ...
├── train.tsv
└── test.tsv
```

TSV format:
```tsv
path	sentence
audio001.mp3	ეს არის ტესტი
audio002.mp3	სხვა წინადადება
```

---

## Configuration

### Edit the Training Script

Open `2_finetune_whisper_local.py` and adjust these settings:

```python
# 1. Choose model based on your hardware
MODEL_NAME = "openai/whisper-small"  # tiny, base, small, medium, large-v3

# 2. Set data paths
DATA_DIR = Path("cv-corpus-23.0-2025-09-05/ka")
CLIPS_DIR = DATA_DIR / "clips"
OUTPUT_DIR = "./whisper-georgian-local"

# 3. Training parameters (optional - script auto-configures)
NUM_EPOCHS = 3           # More epochs = better quality but slower
LEARNING_RATE = 1e-5     # Default is good
```

### Hardware-Specific Configurations

The script automatically detects your hardware and configures:
- **Batch size**: Larger for more VRAM
- **Gradient accumulation**: To simulate larger batches
- **Dataset size**: Limited on low-VRAM or CPU setups

Manual override (if needed):
```python
# Force specific batch size
BATCH_SIZE = 4

# Limit dataset for testing
MAX_TRAIN_SAMPLES = 1000  # Use only 1000 samples
MAX_TEST_SAMPLES = 100
```

---

## Running Training

### Start Training

```bash
# Activate virtual environment
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Run training
python 2_finetune_whisper_local.py
```

### What to Expect

1. **Hardware Detection**: Script shows your GPU/CPU info
2. **Dataset Loading**: Loads and verifies audio files
3. **Model Download**: Downloads Whisper model (first time only)
4. **Preprocessing**: Converts audio to features (takes time)
5. **Training**: Shows progress with loss and WER metrics

### Training Time Estimates

| Hardware | Model | Dataset Size | Estimated Time |
|----------|-------|--------------|----------------|
| RTX 3090 | small | 10,000 samples | ~4-6 hours |
| RTX 3060 | small | 10,000 samples | ~8-12 hours |
| RTX 3060 | tiny | 10,000 samples | ~3-5 hours |
| CPU only | tiny | 1,000 samples | ~24-48 hours |

---

## Monitoring Progress

### TensorBoard (Real-time Monitoring)

Open a new terminal and run:
```bash
tensorboard --logdir ./whisper-georgian-local
```

Then open your browser to: `http://localhost:6006`

You'll see:
- **Loss**: Should decrease over time
- **WER (Word Error Rate)**: Should decrease (lower is better)
- **Learning rate**: Shows training schedule

### Console Output

The script prints:
```
Step 100/3000 | Loss: 0.543 | WER: 45.2%
Step 200/3000 | Loss: 0.421 | WER: 38.7%
...
```

---

## Stopping and Resuming

### Stop Training (Ctrl+C)

The script will save the current model to `whisper-georgian-local_interrupted`

### Resume Training

The Trainer automatically saves checkpoints. To resume:
```python
# In the script, before trainer.train():
trainer.train(resume_from_checkpoint=True)
```

---

## After Training

### Test Your Model

```bash
python transcribe.py --model ./whisper-georgian-local --audio test_audio.mp3
```

### Check Model Quality

Look at the final WER (Word Error Rate):
- **< 10%**: Excellent
- **10-20%**: Good
- **20-30%**: Acceptable
- **> 30%**: Needs more training

---

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms**: `CUDA out of memory` or `RuntimeError: out of memory`

**Solutions**:
1. Reduce batch size:
   ```python
   BATCH_SIZE = 2  # or even 1
   ```

2. Use gradient accumulation:
   ```python
   GRADIENT_ACCUMULATION_STEPS = 8
   ```

3. Use smaller model:
   ```python
   MODEL_NAME = "openai/whisper-tiny"
   ```

4. Limit dataset:
   ```python
   MAX_TRAIN_SAMPLES = 5000
   ```

### Training is Very Slow

**On CPU**:
- This is normal! CPU training is 10-100x slower than GPU
- Reduce dataset size significantly: `MAX_TRAIN_SAMPLES = 500`
- Consider using Google Colab with free GPU

**On GPU**:
- Increase batch size if you have VRAM headroom
- Reduce num_workers if CPU is bottleneck:
  ```python
  NUM_WORKERS = 2
  ```

### CUDA/GPU Not Detected

```bash
# Reinstall PyTorch with CUDA support
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Check NVIDIA driver
nvidia-smi
```

### Dataset Loading Errors

**"File not found"**:
- Verify paths in script match your dataset location
- Check that audio files exist in `clips/` folder

**"No audio samples found"**:
- Ensure TSV file paths are correct
- Check audio file formats (MP3/WAV supported)

### Loss Not Decreasing

- **Too high learning rate**: Try `LEARNING_RATE = 5e-6`
- **Too small dataset**: Need at least 1000+ samples
- **Data quality issues**: Check if transcriptions match audio

---

## Optimizations for Different Scenarios

### Fast Testing (Quick Iteration)

```python
MODEL_NAME = "openai/whisper-tiny"
MAX_TRAIN_SAMPLES = 100
MAX_TEST_SAMPLES = 20
NUM_EPOCHS = 1
```

### Best Quality (Unlimited Time)

```python
MODEL_NAME = "openai/whisper-medium"  # or large-v3
MAX_TRAIN_SAMPLES = None  # Use all data
NUM_EPOCHS = 5
LEARNING_RATE = 5e-6
```

### Limited VRAM (4GB GPU)

```python
MODEL_NAME = "openai/whisper-tiny"
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
USE_GRADIENT_CHECKPOINTING = True
```

### CPU Training (No GPU)

```python
MODEL_NAME = "openai/whisper-tiny"
MAX_TRAIN_SAMPLES = 500
MAX_TEST_SAMPLES = 50
NUM_EPOCHS = 1
BATCH_SIZE = 1
```

---

## Additional Resources

- [Whisper GitHub](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Common Voice Dataset](https://commonvoice.mozilla.org/)
- [TensorBoard Guide](https://www.tensorflow.org/tensorboard)

---

## Quick Reference Commands

```bash
# Install dependencies
pip install torch transformers datasets evaluate jiwer pandas tensorboard

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Run training
python 2_finetune_whisper_local.py

# Monitor training
tensorboard --logdir ./whisper-georgian-local

# Test model
python transcribe.py --model ./whisper-georgian-local --audio test.mp3
```

---

## Need Help?

- Check the error message carefully
- Search for the error on Google/StackOverflow
- Reduce batch size if OOM errors occur
- Start with tiny model and small dataset for testing
