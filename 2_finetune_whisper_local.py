"""
Fine-tune Whisper model LOCALLY on Georgian speech data
Optimized for various hardware configurations (GPU/CPU)
"""

import os
import pandas as pd
import torch
from pathlib import Path
from datasets import Dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

# =============================================================================
# CONFIGURATION - Adjust these based on your hardware
# =============================================================================

# Model selection - choose based on your hardware:
# - "openai/whisper-tiny"   : ~39M params,  ~1GB VRAM,   fastest, lowest accuracy
# - "openai/whisper-base"   : ~74M params,  ~1GB VRAM,   fast, decent accuracy
# - "openai/whisper-small"  : ~244M params, ~2GB VRAM,   good balance
# - "openai/whisper-medium" : ~769M params, ~5GB VRAM,   high accuracy, slower
# - "openai/whisper-large-v3": ~1550M params, ~10GB VRAM, best accuracy, slowest

MODEL_NAME = "openai/whisper-small"  # Change this based on your hardware
LANGUAGE = "ka"  # Georgian
TASK = "transcribe"

# Data paths
DATA_DIR = Path("cv-corpus-23.0-2025-09-05/ka")
CLIPS_DIR = DATA_DIR / "clips"
OUTPUT_DIR = "./whisper-georgian-local"

# =============================================================================
# HARDWARE-SPECIFIC CONFIGURATIONS
# =============================================================================

# Detect available hardware
device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    # GPU available - get VRAM amount
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # Auto-configure based on VRAM
    if gpu_memory_gb >= 20:
        # High-end GPU (RTX 3090, 4090, A100, etc.)
        BATCH_SIZE = 16
        GRADIENT_ACCUMULATION_STEPS = 1
        NUM_WORKERS = 8
        EVAL_BATCH_SIZE = 16
        MAX_TRAIN_SAMPLES = None  # Use all data
        MAX_TEST_SAMPLES = None
    elif gpu_memory_gb >= 10:
        # Mid-range GPU (RTX 3080, 4070 Ti, etc.)
        BATCH_SIZE = 8
        GRADIENT_ACCUMULATION_STEPS = 2
        NUM_WORKERS = 4
        EVAL_BATCH_SIZE = 8
        MAX_TRAIN_SAMPLES = None
        MAX_TEST_SAMPLES = None
    elif gpu_memory_gb >= 6:
        # Entry-level GPU (RTX 3060, etc.)
        BATCH_SIZE = 4
        GRADIENT_ACCUMULATION_STEPS = 4
        NUM_WORKERS = 2
        EVAL_BATCH_SIZE = 4
        MAX_TRAIN_SAMPLES = 10000  # Limit dataset size
        MAX_TEST_SAMPLES = 1000
    else:
        # Low VRAM GPU
        BATCH_SIZE = 2
        GRADIENT_ACCUMULATION_STEPS = 8
        NUM_WORKERS = 2
        EVAL_BATCH_SIZE = 2
        MAX_TRAIN_SAMPLES = 5000
        MAX_TEST_SAMPLES = 500
else:
    # CPU only - very slow but will work
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 16
    NUM_WORKERS = 2
    EVAL_BATCH_SIZE = 1
    MAX_TRAIN_SAMPLES = 1000  # Strongly limit for CPU
    MAX_TEST_SAMPLES = 100

# Training hyperparameters
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 500
SAVE_STEPS = 500
EVAL_STEPS = 500
LOGGING_STEPS = 25

# Memory optimization flags
USE_GRADIENT_CHECKPOINTING = True  # Reduces VRAM at cost of speed
USE_FP16 = torch.cuda.is_available()  # Use mixed precision on GPU


def prepare_dataset(max_train_samples=None, max_test_samples=None):
    """Load and prepare the Common Voice Georgian dataset"""
    print("Loading data...")

    # Load TSV files
    train_df = pd.read_csv(DATA_DIR / "train.tsv", sep='\t')
    test_df = pd.read_csv(DATA_DIR / "test.tsv", sep='\t')

    # Prepare data: keep only path and sentence
    def prepare_df(df, max_samples=None):
        # Create full audio path
        df['audio'] = df['path'].apply(lambda x: str(CLIPS_DIR / x))
        # Filter: only keep rows where audio file exists
        df = df[df['audio'].apply(lambda x: os.path.exists(x))]
        # Limit samples if specified
        if max_samples is not None and len(df) > max_samples:
            df = df.head(max_samples)
            print(f"  Limited to {max_samples} samples")
        # Keep only audio path and sentence
        return df[['audio', 'sentence']].rename(columns={'sentence': 'transcription'})

    train_prepared = prepare_df(train_df, max_train_samples)
    test_prepared = prepare_df(test_df, max_test_samples)

    print(f"Training samples: {len(train_prepared)}")
    print(f"Test samples: {len(test_prepared)}")

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_prepared)
    test_dataset = Dataset.from_pandas(test_prepared)

    # Cast audio column to Audio type
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

    return DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })


def prepare_data_for_training(batch, feature_extractor, tokenizer):
    """Prepare audio and text data for training"""
    # Load and resample audio data to 16kHz
    audio = batch["audio"]

    # Compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Encode target text to label ids
    batch["labels"] = tokenizer(batch["transcription"]).input_ids

    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for speech-to-text"""
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred, tokenizer, metric):
    """Compute WER metric"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def print_hardware_info():
    """Print detailed hardware information"""
    print("=" * 70)
    print("HARDWARE CONFIGURATION")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"Device: GPU ({torch.cuda.get_device_name(0)})")
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Memory: {gpu_mem_gb:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("Device: CPU")
        print("WARNING: No GPU detected! Training will be VERY slow.")
        print("Consider using Google Colab or reducing dataset size significantly.")

    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"CPU Cores: {os.cpu_count()}")


def print_training_config():
    """Print training configuration"""
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Language: {LANGUAGE}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"\nBatch size (per device): {BATCH_SIZE}")
    print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Evaluation batch size: {EVAL_BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"FP16 (Mixed Precision): {USE_FP16}")
    print(f"Gradient Checkpointing: {USE_GRADIENT_CHECKPOINTING}")
    print(f"Data loading workers: {NUM_WORKERS}")

    if MAX_TRAIN_SAMPLES:
        print(f"\nMax training samples: {MAX_TRAIN_SAMPLES}")
        print(f"Max test samples: {MAX_TEST_SAMPLES}")
    else:
        print(f"\nUsing full dataset")


def main():
    print("=" * 70)
    print("WHISPER LOCAL FINE-TUNING FOR GEORGIAN SPEECH RECOGNITION")
    print("=" * 70)

    # Print hardware info
    print_hardware_info()

    # Print training config
    print_training_config()

    # Load dataset
    print("\n" + "=" * 70)
    print("STEP 1: PREPARING DATASET")
    print("=" * 70)
    dataset = prepare_dataset(MAX_TRAIN_SAMPLES, MAX_TEST_SAMPLES)

    # Load Whisper model components
    print("\n" + "=" * 70)
    print("STEP 2: LOADING MODEL COMPONENTS")
    print("=" * 70)
    print(f"Downloading {MODEL_NAME}...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    print("Model components loaded successfully!")

    # Prepare datasets
    print("\n" + "=" * 70)
    print("STEP 3: PREPROCESSING AUDIO DATA")
    print("=" * 70)
    print("This may take several minutes...")
    dataset = dataset.map(
        lambda batch: prepare_data_for_training(batch, feature_extractor, tokenizer),
        remove_columns=dataset.column_names["train"],
        num_proc=NUM_WORKERS
    )
    print("Preprocessing complete!")

    # Load pre-trained model
    print("\n" + "=" * 70)
    print("STEP 4: LOADING PRE-TRAINED MODEL")
    print("=" * 70)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    print("Model loaded!")

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Metric
    metric = evaluate.load("wer")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        fp16=USE_FP16,
        evaluation_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        logging_steps=LOGGING_STEPS,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        dataloader_num_workers=NUM_WORKERS,
        # Save disk space
        save_total_limit=2,  # Keep only 2 checkpoints
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer, metric),
        tokenizer=processor.feature_extractor,
    )

    # Train
    print("\n" + "=" * 70)
    print("STEP 5: TRAINING")
    print("=" * 70)
    print(f"Training samples: {len(dataset['train'])}")
    print(f"Eval samples: {len(dataset['test'])}")
    print(f"Total steps: ~{len(dataset['train']) * NUM_EPOCHS // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)}")
    print("\nTraining started! This will take a while...")
    print("You can monitor progress with TensorBoard:")
    print(f"  tensorboard --logdir {OUTPUT_DIR}")
    print("")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current model state...")
        trainer.save_model(OUTPUT_DIR + "_interrupted")
        processor.save_pretrained(OUTPUT_DIR + "_interrupted")
        print(f"Model saved to {OUTPUT_DIR}_interrupted")
        return

    # Save final model
    print("\n" + "=" * 70)
    print("STEP 6: SAVING MODEL")
    print("=" * 70)
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Model saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Test your model:")
    print(f"   python transcribe.py --model {OUTPUT_DIR} --audio your_audio.mp3")
    print("\n2. View training metrics:")
    print(f"   tensorboard --logdir {OUTPUT_DIR}")
    print("\n3. Optionally push to Hugging Face Hub for sharing")


if __name__ == "__main__":
    main()
