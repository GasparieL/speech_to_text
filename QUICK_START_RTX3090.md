# Quick Start Guide - RTX 3090 Training

## TL;DR - What You Need to Do

### On Your Mac (Right Now):

1. **Package everything for transfer:**
   ```bash
   cd /Users/lana/Desktop/speech_to_text
   ./package_for_training.sh
   ```
   This creates a `package_transfer/` folder with everything you need.

2. **Transfer to your RTX 3090 machine** (choose one method):

   **Option A - Direct transfer via network:**
   ```bash
   scp -r package_transfer/ user@your-rtx-machine:/path/to/training/
   ```

   **Option B - USB drive:**
   - Copy `package_transfer/` folder to USB drive
   - Plug into training machine

   **Option C - Cloud storage:**
   - Upload `package_transfer/` to Google Drive/Dropbox
   - Download on training machine

### On Your RTX 3090 Machine:

1. **Extract and setup:**
   ```bash
   cd /path/to/training/package_transfer
   tar -xzf georgian_dataset.tar.gz

   # Install dependencies
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   ```

2. **Verify GPU:**
   ```bash
   python -c "import torch; print('CUDA:', torch.cuda.is_available(), '- GPU:', torch.cuda.get_device_name(0))"
   ```
   Should show: `CUDA: True - GPU: NVIDIA GeForce RTX 3090`

3. **Start training:**
   ```bash
   python 2_finetune_whisper.py
   ```

4. **Wait 8-12 hours** ☕

5. **Copy model back to Mac:**
   ```bash
   tar -czf whisper-georgian-finetuned.tar.gz whisper-georgian-finetuned/
   # Transfer back to Mac (same methods as before)
   ```

### Back on Your Mac:

1. **Extract and use:**
   ```bash
   tar -xzf whisper-georgian-finetuned.tar.gz

   # Transcribe audio
   python 3_transcribe.py --audio my_audio.mp3

   # Or use web interface
   python 4_gradio_app.py
   ```

## Training Time Guide

| Model Choice | Edit Line 23 in Script | Time | Accuracy |
|--------------|------------------------|------|----------|
| Quick test | `"openai/whisper-tiny"` | 2-3h | Good |
| Fast | `"openai/whisper-base"` | 4-6h | Better |
| **Recommended** | `"openai/whisper-small"` | **8-12h** | **Excellent** |
| Best quality | `"openai/whisper-medium"` | 20-30h | Best |

## What Files You're Transferring

```
package_transfer/
├── georgian_dataset.tar.gz    # Your 146 hours of audio + transcripts
├── 2_finetune_whisper.py      # Training script (RTX 3090 optimized)
├── 3_transcribe.py            # Use the model after training
├── 4_gradio_app.py            # Web interface
├── requirements.txt           # Python packages
└── README.md                  # Full setup guide
```

Total size: ~7 GB

## Need Help?

See [SETUP_RTX3090.md](SETUP_RTX3090.md) for detailed instructions and troubleshooting.

## Expected Results

Before fine-tuning (what you saw earlier):
```
"Kaisat marvel寒čenelyajik es wireco-alida Minecraft BA"
```

After fine-tuning on RTX 3090:
```
"ის პარიზში ცხოვრობდა ცნობილ ვარსკვლავ სტელასთან ერთად"
```

Much better! 🎉
