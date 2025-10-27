# Training Setup for RTX 3090

Perfect! With an RTX 3090, you can train efficiently on your local machine.

## ⚡ Expected Training Times (RTX 3090)

| Model | Training Time | Accuracy | Recommended |
|-------|---------------|----------|-------------|
| whisper-tiny | 2-3 hours | Good | Quick testing |
| whisper-base | 4-6 hours | Better | Testing |
| **whisper-small** | **8-12 hours** | **Excellent** | ⭐ **Best choice** |
| whisper-medium | 20-30 hours | Best | If you have time |
| whisper-large-v3 | 40-50 hours | Best+ | Maximum accuracy |

## 🚀 Setup Instructions

### Step 1: Transfer Data to Your RTX 3090 Machine

You need to copy the dataset from your Mac to your training machine:

```bash
# On your Mac, create a tarball of just what you need:
cd /Users/lana/Desktop/speech_to_text
tar -czf georgian_dataset.tar.gz cv-corpus-23.0-2025-09-05/

# Transfer to your training machine (adjust path/IP):
scp georgian_dataset.tar.gz user@your-machine:/path/to/training/
scp -r /Users/lana/Desktop/speech_to_text/*.py user@your-machine:/path/to/training/
scp requirements.txt user@your-machine:/path/to/training/
```

### Step 2: On Your RTX 3090 Machine

```bash
# Extract dataset
tar -xzf georgian_dataset.tar.gz

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate evaluate jiwer librosa soundfile gradio pandas numpy
```

### Step 3: Verify GPU is Detected

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
```

### Step 4: Choose Your Model Size

Edit `2_finetune_whisper.py` line 23:

```python
# For fastest training (2-3 hours):
MODEL_NAME = "openai/whisper-tiny"

# For good balance (4-6 hours):
MODEL_NAME = "openai/whisper-base"

# For best results (8-12 hours) - RECOMMENDED:
MODEL_NAME = "openai/whisper-small"

# For maximum accuracy (20-30 hours):
MODEL_NAME = "openai/whisper-medium"
```

### Step 5: Start Training

```bash
python 2_finetune_whisper.py
```

### Step 6: Monitor Training

Open another terminal and run:
```bash
tensorboard --logdir ./whisper-georgian-finetuned/runs
```

Then open http://localhost:6006 in your browser to see:
- Training loss
- Validation loss
- Word Error Rate (WER)

## ⚙️ RTX 3090 Optimizations

The script is already optimized for 3090 (24GB VRAM):

| Model Size | Batch Size | Memory Usage | Speed |
|------------|------------|--------------|-------|
| tiny       | 64         | ~8 GB        | Fastest |
| base       | 48         | ~12 GB       | Very Fast |
| small      | 32         | ~18 GB       | Fast |
| medium     | 16         | ~22 GB       | Good |
| large-v3   | 8-12       | ~23 GB       | Slower |

Current settings in the script:
```python
BATCH_SIZE = 32  # Perfect for whisper-small on 3090
```

### If you get Out of Memory (OOM) errors:

Edit line 32 in `2_finetune_whisper.py`:
```python
BATCH_SIZE = 16  # or 8
GRADIENT_ACCUMULATION_STEPS = 2  # Maintains effective batch size
```

## 📊 What to Expect

### During Training:
```
Step 100: loss=0.52, wer=45.2%
Step 500: loss=0.31, wer=28.5%
Step 1000: loss=0.19, wer=18.3%
Step 2000: loss=0.12, wer=12.7%
...
```

### Final Results:
- **Pre-trained Whisper WER**: ~40-60% (basically random)
- **After fine-tuning**: ~10-20% WER
- **Improvement**: 3-4x more accurate!

## 💾 After Training

The model will be saved to `./whisper-georgian-finetuned/`

### Copy back to Mac for inference:

```bash
# On training machine
cd /path/to/training
tar -czf whisper-georgian-finetuned.tar.gz whisper-georgian-finetuned/

# Copy to Mac
scp whisper-georgian-finetuned.tar.gz user@your-mac:/Users/lana/Desktop/speech_to_text/
```

### Use on Mac:

```bash
# Extract
tar -xzf whisper-georgian-finetuned.tar.gz

# Transcribe
python 3_transcribe.py --audio your_audio.mp3

# Or use web interface
python 4_gradio_app.py
```

## 🎯 Recommended Workflow

1. **Start with whisper-small** (8-12 hours)
2. **Evaluate results** on test set
3. **If accuracy is good enough**: Done! ✓
4. **If you need better**: Train whisper-medium (20-30 hours)

## 🔧 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size in 2_finetune_whisper.py
BATCH_SIZE = 16  # or 8
```

### Training is slow
- Check GPU usage: `nvidia-smi`
- Should see ~95%+ GPU utilization
- If low, increase `NUM_WORKERS`

### Loss not decreasing
- Normal for first few hundred steps
- Should start decreasing after ~500 steps
- If stuck, try increasing learning rate to 2e-5

## 📈 Tips for Best Results

1. **Let it train for all 3 epochs** - Don't stop early
2. **Monitor WER** - Should decrease steadily
3. **Save checkpoints** - Already configured every 1000 steps
4. **Test on real audio** - After training, test with your own recordings

## Next Steps

After training completes:
1. Test the model: `python 3_transcribe.py --audio test.mp3`
2. Launch web interface: `python 4_gradio_app.py`
3. Celebrate! You now have a Georgian speech-to-text model! 🎉
