# Georgian Speech-to-Text with Whisper

Fine-tune OpenAI's Whisper model on Georgian speech data for accurate transcription.

## Dataset
- **Source**: Mozilla Common Voice Georgian (v23.0)
- **Size**: ~146 hours of speech
- **Structure**:
  - Audio files in `cv-corpus-23.0-2025-09-05/ka/clips/`
  - Transcriptions in `train.tsv`, `test.tsv`, `dev.tsv`

## Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Explore the Data
Run the data preparation notebook to understand your dataset:
```bash
jupyter notebook 1_prepare_data.ipynb
```

### Step 3: Fine-tune Whisper
```bash
python 2_finetune_whisper.py
```

**Training Configuration:**
- Model: `whisper-small` (you can change to `tiny`, `base`, `medium`, or `large-v3`)
- Epochs: 3
- Batch size: 16 (reduce if you get out of memory errors)
- Learning rate: 1e-5

**Training time estimates:**
- With GPU (RTX 3090): ~10-15 hours
- With GPU (T4): ~20-30 hours
- Without GPU: Not recommended (will take days)

The model will be saved to `./whisper-georgian-finetuned/`

### Step 4: Transcribe Audio

**Command Line:**
```bash
python 3_transcribe.py --audio path/to/your/audio.mp3
```

**Web Interface:**
```bash
python 4_gradio_app.py
```
Then open http://localhost:7860 in your browser.

To create a public link:
```bash
python 4_gradio_app.py --share
```

## Usage Examples

### Transcribe a single file
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# Load model
processor = WhisperProcessor.from_pretrained("./whisper-georgian-finetuned")
model = WhisperForConditionalGeneration.from_pretrained("./whisper-georgian-finetuned")

# Load audio
audio, sr = librosa.load("audio.mp3", sr=16000)

# Transcribe
input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print(transcription)
```

### Batch transcription
```python
import os
from pathlib import Path

audio_dir = Path("my_audio_files")
for audio_file in audio_dir.glob("*.mp3"):
    transcription = transcribe_audio(audio_file, model, processor)
    print(f"{audio_file.name}: {transcription}")
```

## Configuration

### Model Size Options
Edit `MODEL_NAME` in `2_finetune_whisper.py`:
- `openai/whisper-tiny` - Fastest, least accurate (~39M params)
- `openai/whisper-base` - Fast, good for testing (~74M params)
- `openai/whisper-small` - **Recommended** balance (~244M params)
- `openai/whisper-medium` - More accurate (~769M params)
- `openai/whisper-large-v3` - Best accuracy (~1550M params)

### Training Parameters
Adjust these in `2_finetune_whisper.py`:
- `BATCH_SIZE`: Reduce if GPU memory errors (try 8, 4, or 2)
- `NUM_EPOCHS`: More epochs = better accuracy (but risk overfitting)
- `LEARNING_RATE`: Lower = more stable, higher = faster convergence

### Reduce Memory Usage
If you get out of memory errors:
1. Reduce `BATCH_SIZE` to 8 or 4
2. Use smaller model (`whisper-tiny` or `whisper-base`)
3. Increase `GRADIENT_ACCUMULATION_STEPS` to 2 or 4

## Monitoring Training

View training progress with TensorBoard:
```bash
tensorboard --logdir ./whisper-georgian-finetuned/runs
```

## Troubleshooting

### Out of Memory
```python
# In 2_finetune_whisper.py, reduce batch size:
BATCH_SIZE = 4  # or even 2
GRADIENT_ACCUMULATION_STEPS = 4
```

### Audio files not found
Check that the path to `cv-corpus-23.0-2025-09-05/ka/clips/` is correct.

### Low accuracy
- Train for more epochs
- Use a larger model (`medium` or `large-v3`)
- Ensure your audio quality is good

## Expected Results

After fine-tuning on 146 hours of Georgian speech:
- **Word Error Rate (WER)**: 10-20% (depends on model size)
- Pre-trained Whisper without fine-tuning: ~30-40% WER
- Your fine-tuned model should be **2-3x more accurate**!

## Next Steps

1. **Test the pre-trained model first** (before fine-tuning):
   ```python
   import whisper
   model = whisper.load_model("small")
   result = model.transcribe("audio.mp3", language="ka")
   print(result["text"])
   ```

2. **Fine-tune on your data** to improve accuracy

3. **Evaluate on your test set** to measure improvement

4. **Deploy** using the Gradio interface or API

## Files Overview

- `requirements.txt` - Python dependencies
- `1_prepare_data.ipynb` - Explore and prepare dataset
- `2_finetune_whisper.py` - Main training script
- `3_transcribe.py` - Command-line transcription
- `4_gradio_app.py` - Web interface for transcription

## Contributing

Feel free to modify the scripts for your needs!

## License

This project uses OpenAI's Whisper model and Mozilla Common Voice data.
Check their respective licenses for usage terms.
# speech_to_text
