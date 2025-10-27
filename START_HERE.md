# Georgian Speech-to-Text - START HERE

## 🎯 Overview

You have **146 hours of Georgian speech + transcriptions** and want to build a speech-to-text system.

**Solution:** Fine-tune OpenAI's Whisper model on your data.

## 📦 What You Have

✓ Dataset: Mozilla Common Voice Georgian v23.0 (227,211 audio files)
✓ Training files created and optimized for RTX 3090
✓ All scripts ready to go

## ⚡ You Have an RTX 3090!

**Great news!** Training will take **8-12 hours** (for whisper-small model).

## 🚀 Quick Start (3 Steps)

### Step 1: Package Everything (On Mac)
```bash
cd /Users/lana/Desktop/speech_to_text
./package_for_training.sh
```

### Step 2: Transfer to RTX 3090 Machine
Transfer the `package_transfer/` folder to your training machine.

### Step 3: Train (On RTX 3090)
```bash
cd package_transfer/
tar -xzf georgian_dataset.tar.gz
pip install -r requirements.txt
python 2_finetune_whisper.py
```

**That's it!** Come back in 8-12 hours.

## 📚 Files Guide

| File | Purpose |
|------|---------|
| **START_HERE.md** | This file - overview |
| **QUICK_START_RTX3090.md** | Step-by-step guide |
| **SETUP_RTX3090.md** | Detailed setup & troubleshooting |
| [2_finetune_whisper.py](2_finetune_whisper.py) | Training script (RTX 3090 optimized) |
| [3_transcribe.py](3_transcribe.py) | Command-line transcription |
| [4_gradio_app.py](4_gradio_app.py) | Web interface for upload & transcribe |
| [0_quick_test.py](0_quick_test.py) | Test pre-trained model (no training) |
| [requirements.txt](requirements.txt) | Python dependencies |
| package_for_training.sh | Packages everything for transfer |

## ⏱️ Timeline

1. **Package files**: 5-10 minutes
2. **Transfer to RTX 3090**: 10-30 minutes (depends on method)
3. **Setup on RTX 3090**: 5 minutes
4. **Training**: 8-12 hours
5. **Transfer model back**: 5 minutes
6. **Ready to use!**

## 🎓 What You'll Get

**Before fine-tuning** (pre-trained Whisper):
- Word Error Rate: ~40-60%
- Output: Gibberish (as you saw: "Kaisat marvel寒čenelyajik...")

**After fine-tuning** (your model):
- Word Error Rate: ~10-20%
- Output: Accurate Georgian transcription!
- **3-4x improvement!**

## 💡 Model Size Guide

Edit line 23 in [2_finetune_whisper.py:23](2_finetune_whisper.py#L23):

```python
# Pick one:
MODEL_NAME = "openai/whisper-tiny"    # 2-3h training, good accuracy
MODEL_NAME = "openai/whisper-base"    # 4-6h training, better accuracy
MODEL_NAME = "openai/whisper-small"   # 8-12h training, excellent ⭐
MODEL_NAME = "openai/whisper-medium"  # 20-30h training, best accuracy
```

**Recommendation:** Start with `whisper-small` for best balance.

## 🖥️ After Training - Using the Model

### Command line:
```bash
python 3_transcribe.py --audio my_georgian_audio.mp3
```

### Web interface:
```bash
python 4_gradio_app.py
```
Opens at http://localhost:7860 - upload audio, get transcription!

### Python code:
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

processor = WhisperProcessor.from_pretrained("./whisper-georgian-finetuned")
model = WhisperForConditionalGeneration.from_pretrained("./whisper-georgian-finetuned")

audio, _ = librosa.load("audio.mp3", sr=16000)
input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print(transcription)
```

## 🆘 Need Help?

1. **Setup questions?** → See [QUICK_START_RTX3090.md](QUICK_START_RTX3090.md)
2. **Detailed setup?** → See [SETUP_RTX3090.md](SETUP_RTX3090.md)
3. **Problems during training?** → Check "Troubleshooting" section in SETUP_RTX3090.md

## ✅ Ready to Start?

**Run this now on your Mac:**
```bash
cd /Users/lana/Desktop/speech_to_text
./package_for_training.sh
```

Then follow [QUICK_START_RTX3090.md](QUICK_START_RTX3090.md)!

Good luck! 🚀
