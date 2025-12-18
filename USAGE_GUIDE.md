# Quick Usage Guide

## Overview

This guide shows you exactly what to run and when. For detailed explanations, see [DETAILED_EXPLANATION.md](DETAILED_EXPLANATION.md).

---

## Part 1: Training (GPU Computer)

### Prerequisites
- NVIDIA GPU with 6GB+ VRAM (or CPU, but much slower)
- Python 3.8+
- 50GB+ free disk space

### Setup (One-time)

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 3. Install PyTorch with GPU support
# For NVIDIA GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CPU only (slower):
pip install torch torchvision torchaudio

# 4. Install other dependencies
pip install transformers datasets evaluate jiwer pandas tensorboard librosa soundfile gradio

# 5. Verify GPU is detected (skip if using CPU)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### Prepare Dataset

1. Download Common Voice Georgian dataset from: https://commonvoice.mozilla.org/en/datasets
2. Extract to project folder so you have:
   ```
   cv-corpus-23.0-2025-09-05/ka/
   ├── clips/
   ├── train.tsv
   └── test.tsv
   ```

### Run Training

```bash
# Make sure venv is activated
python 2_finetune_whisper_local.py
```

**Expected time:** 4-12 hours depending on GPU

**Monitor progress:**
```bash
# In another terminal
tensorboard --logdir ./whisper-georgian-local
# Open http://localhost:6006 in browser
```

**Result:** `whisper-georgian-local/` folder containing your trained model

---

## Part 2: Using the Model

Once training is complete, you have 3 options to transcribe audio:

### Option A: Command Line (Single File)

```bash
# Activate environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Transcribe single audio file
python 5_transcribe_to_file.py --model ./whisper-georgian-local --audio my_audio.mp3

# Result: Creates my_audio_transcription.txt
```

**Supported formats:** MP3, WAV, M4A, FLAC, OGG

### Option B: Command Line (Batch Processing)

```bash
# Transcribe all audio files in a folder
python 5_transcribe_to_file.py --model ./whisper-georgian-local --audio-folder ./my_podcasts/

# Result: Creates .txt file for each audio file in the same folder
```

**With custom output folder:**
```bash
python 5_transcribe_to_file.py --model ./whisper-georgian-local --audio-folder ./podcasts/ --output-folder ./transcriptions/
```

### Option C: Web Interface (User-Friendly)

```bash
# Start web server
python 6_web_interface.py --model ./whisper-georgian-local

# Open http://localhost:7860 in your browser
```

**Features:**
- Drag and drop audio files
- Record directly from microphone
- View transcription in real-time
- Download as .txt file with one click

**Access from other devices on network:**
```bash
python 6_web_interface.py --model ./whisper-georgian-local --server-name 0.0.0.0
# Access from other devices using: http://YOUR_IP:7860
```

**Create public link (accessible from internet):**
```bash
python 6_web_interface.py --model ./whisper-georgian-local --share
# Gradio will generate a public URL you can share
```

---

## Part 3: Transfer Model to Another Computer

### On GPU Computer (after training)

```bash
# Compress the trained model
zip -r whisper-georgian-local.zip whisper-georgian-local/

# Or use tar
tar -czf whisper-georgian-local.tar.gz whisper-georgian-local/
```

### Transfer
- Copy via USB drive
- Upload to Google Drive / Dropbox
- Use SCP if on same network: `scp -r whisper-georgian-local/ user@other-pc:/path/`

### On Destination Computer

```bash
# 1. Extract
unzip whisper-georgian-local.zip
# Or: tar -xzf whisper-georgian-local.tar.gz

# 2. Install minimal dependencies (CPU inference only needs fewer packages)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install torch transformers librosa gradio

# 3. Use the model
python 5_transcribe_to_file.py --model ./whisper-georgian-local --audio test.mp3
# Or start web interface
python 6_web_interface.py --model ./whisper-georgian-local
```

---

## Common Commands Cheat Sheet

### Training
```bash
# Start training
python 2_finetune_whisper_local.py

# Monitor training
tensorboard --logdir ./whisper-georgian-local
```

### Transcription
```bash
# Single file
python 5_transcribe_to_file.py --model ./whisper-georgian-local --audio file.mp3

# Batch processing
python 5_transcribe_to_file.py --model ./whisper-georgian-local --audio-folder ./folder/

# Web interface (local)
python 6_web_interface.py --model ./whisper-georgian-local

# Web interface (public link)
python 6_web_interface.py --model ./whisper-georgian-local --share
```

### Help
```bash
# Get help for any script
python 5_transcribe_to_file.py --help
python 6_web_interface.py --help
python 2_finetune_whisper_local.py --help
```

---

## Output File Format

When you transcribe audio, you get a `.txt` file with this format:

```
Georgian Speech Transcription
============================================================

Generated: 2024-01-15 14:32:05
Model: whisper-georgian-local
Audio File: my_recording.mp3
Duration: 3m 45s

────────────────────────────────────────────────────────────

გამარჯობა, ეს არის ჩანაწერის ტრანსკრიფცია.
აქ იქნება მთელი ტექსტი რომელიც თქვა ხმოვან ჩანაწერში.

────────────────────────────────────────────────────────────

Transcription completed successfully.
```

---

## Troubleshooting

### "CUDA out of memory"
```python
# Edit 2_finetune_whisper_local.py:
MODEL_NAME = "openai/whisper-tiny"  # Use smaller model
BATCH_SIZE = 2  # Reduce batch size
```

### "Model not found"
```bash
# Make sure path is correct
ls -la whisper-georgian-local/
# Should see: config.json, model.safetensors, etc.

# If missing, retrain:
python 2_finetune_whisper_local.py
```

### "No module named 'torch'"
```bash
# Activate virtual environment
source venv/bin/activate  # You should see (venv) in prompt

# Reinstall dependencies
pip install torch transformers librosa gradio
```

### Training is very slow
```bash
# Check GPU is being used
nvidia-smi
# Should show Python process using GPU

# If using CPU, reduce dataset:
# Edit 2_finetune_whisper_local.py:
MAX_TRAIN_SAMPLES = 1000
MODEL_NAME = "openai/whisper-tiny"
```

### Web interface won't start
```bash
# Try different port
python 6_web_interface.py --model ./whisper-georgian-local --port 8080

# Check if model exists
ls whisper-georgian-local/
```

---

## Performance Expectations

### Training Time (on GPU)
| GPU | Model | Dataset | Time |
|-----|-------|---------|------|
| RTX 4090 | small | 10k samples | ~3-4h |
| RTX 3090 | small | 10k samples | ~4-6h |
| RTX 3060 | small | 10k samples | ~8-12h |
| RTX 3060 | tiny | 10k samples | ~3-5h |

### Inference Speed
| Hardware | Model | Speed |
|----------|-------|-------|
| RTX 3090 | small | ~20x real-time |
| RTX 3060 | small | ~8x real-time |
| CPU (modern) | tiny | ~1x real-time |

*Speed: 10x real-time means 1 minute of audio processes in 6 seconds*

### Quality (WER - Word Error Rate)
- **< 10%**: Excellent
- **10-20%**: Good
- **20-30%**: Acceptable
- **> 30%**: Needs improvement

---

## File Structure

After setup and training, your project should look like:

```
speech_to_text/
├── venv/                              # Virtual environment
├── cv-corpus-23.0-2025-09-05/ka/     # Training data
│   ├── clips/
│   ├── train.tsv
│   └── test.tsv
├── whisper-georgian-local/           # Trained model
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── 2_finetune_whisper_local.py       # Training script
├── 5_transcribe_to_file.py           # CLI transcription
├── 6_web_interface.py                # Web interface
├── DETAILED_EXPLANATION.md           # Detailed docs
├── USAGE_GUIDE.md                    # This file
└── LOCAL_TRAINING_GUIDE.md           # Training guide
```

---

## Quick Start Summary

**For Training:**
1. Install dependencies: `pip install torch transformers datasets evaluate jiwer pandas tensorboard`
2. Download dataset from Common Voice
3. Run: `python 2_finetune_whisper_local.py`
4. Wait 4-12 hours

**For Using:**
1. Install dependencies: `pip install torch transformers librosa gradio`
2. Run: `python 6_web_interface.py --model ./whisper-georgian-local`
3. Open http://localhost:7860
4. Upload audio, get text file!

---

## Need More Help?

- **Detailed explanations:** Read [DETAILED_EXPLANATION.md](DETAILED_EXPLANATION.md)
- **Training setup:** Read [LOCAL_TRAINING_GUIDE.md](LOCAL_TRAINING_GUIDE.md)
- **Error messages:** Check the error carefully and search online
- **GPU issues:** Run `nvidia-smi` to verify GPU is working

---

## Tips for Best Results

### Training
- Use at least 5,000 training samples
- Train for 3-5 epochs
- Use the largest model your GPU can handle
- Monitor WER - stop if it stops improving

### Transcription
- Use clear audio with minimal background noise
- Avoid music playing in background
- Better microphone = better results
- Split very long files into shorter segments

### Hardware
- GPU is highly recommended for training
- CPU is fine for inference/transcription
- 16GB+ RAM recommended
- SSD storage recommended

---

**That's it! You're ready to use Georgian speech-to-text.**
