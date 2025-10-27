# Training Whisper on Mac (Apple Silicon or Intel)

## ⚠️ Reality Check

**Training on Mac is SLOW but POSSIBLE:**
- **whisper-tiny**: ~1-2 days
- **whisper-base**: ~2-3 days
- **whisper-small**: ~5-7 days ❌ Not recommended
- **whisper-medium/large**: 10+ days ❌❌ Don't even try

**Recommended approach:** Use Google Colab (free GPU) - see `COLAB_TRAINING.ipynb`

## If you REALLY want to train on Mac:

### 1. Install Dependencies (MPS support for Apple Silicon)

```bash
# Install PyTorch with MPS support (for M1/M2/M3 Macs)
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Install other requirements
pip3 install transformers datasets accelerate evaluate jiwer librosa soundfile gradio pandas
```

### 2. Modify the Training Script for Mac

Create a new file `2_finetune_whisper_mac.py`:

```python
# At the top, change device detection:
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Change training arguments:
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,  # Reduced for Mac
    gradient_accumulation_steps=4,  # Accumulate gradients
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    num_train_epochs=NUM_EPOCHS,
    gradient_checkpointing=True,
    fp16=False,  # ← Changed! Mac doesn't support fp16
    evaluation_strategy="steps",
    per_device_eval_batch_size=4,  # Reduced
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    dataloader_num_workers=0,  # ← Added for Mac compatibility
)
```

### 3. Use Smaller Model and Less Data

```python
# Use tiny model
MODEL_NAME = "openai/whisper-tiny"  # Much faster!

# Or use subset of data for testing
train_prepared = prepare_df(train_df).head(1000)  # Only 1000 samples
```

### 4. Run Training

```bash
python 2_finetune_whisper_mac.py
```

**Monitor progress:**
```bash
# Open another terminal
tensorboard --logdir ./whisper-georgian-finetuned/runs
```

### 5. Mac-Specific Tips

**Prevent sleep during training:**
```bash
caffeinate -i python 2_finetune_whisper_mac.py
```

**Monitor temperature:**
- Your Mac will get HOT
- Use a cooling pad
- Keep vents clear
- Consider training overnight

**Memory management:**
- Close all other apps
- 16GB RAM minimum
- 32GB RAM recommended

## Better Alternative: Google Colab

1. Upload `COLAB_TRAINING.ipynb` to Google Colab
2. Upload your dataset to Google Drive (or upload the tar.gz)
3. Run all cells
4. Download the trained model
5. Use it on your Mac for inference (fast!)

**Colab advantages:**
- ✅ Free GPU (T4)
- ✅ 10-15 hours vs 5-7 days
- ✅ No battery drain
- ✅ No overheating
- ✅ Can close your laptop

## Just Want to Test First?

You can use pre-trained Whisper on Mac (no fine-tuning) RIGHT NOW:

```python
import whisper

model = whisper.load_model("small")
result = model.transcribe("audio.mp3", language="ka")
print(result["text"])
```

This works fine on Mac! Fine-tuning is what's slow, but inference is fast enough.

## Recommendation

1. **Test pre-trained Whisper on Mac** (5 minutes)
2. **Fine-tune on Google Colab** (10-15 hours)
3. **Download the model and use on Mac** (inference is fast!)

This gives you the best of both worlds!
