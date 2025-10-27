"""
Fine-tune Whisper model on Georgian speech data
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

# Configuration
MODEL_NAME = "openai/whisper-small"  # Options: tiny, base, small, medium, large-v3
LANGUAGE = "ka"  # Georgian
TASK = "transcribe"

DATA_DIR = Path("cv-corpus-23.0-2025-09-05/ka")
CLIPS_DIR = DATA_DIR / "clips"
OUTPUT_DIR = "./whisper-georgian-finetuned"

# Training parameters (Optimized for RTX 3090 - 24GB VRAM)
BATCH_SIZE = 32  # 3090 can handle larger batches
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 500
SAVE_STEPS = 1000
EVAL_STEPS = 1000
NUM_WORKERS = 4  # For faster data loading


def prepare_dataset():
    """Load and prepare the Common Voice Georgian dataset"""
    print("Loading data...")

    # Load TSV files
    train_df = pd.read_csv(DATA_DIR / "train.tsv", sep='\t')
    test_df = pd.read_csv(DATA_DIR / "test.tsv", sep='\t')

    # Prepare data: keep only path and sentence
    def prepare_df(df):
        # Create full audio path
        df['audio'] = df['path'].apply(lambda x: str(CLIPS_DIR / x))
        # Filter: only keep rows where audio file exists
        df = df[df['audio'].apply(lambda x: os.path.exists(x))]
        # Keep only audio path and sentence
        return df[['audio', 'sentence']].rename(columns={'sentence': 'transcription'})

    train_prepared = prepare_df(train_df)
    test_prepared = prepare_df(test_df)

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


def main():
    print("=" * 50)
    print("Whisper Fine-tuning for Georgian Speech Recognition")
    print("=" * 50)

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  WARNING: No GPU detected! Training will be VERY slow.")

    # Load dataset
    print("\n1. Preparing dataset...")
    dataset = prepare_dataset()

    # Load Whisper model components
    print(f"\n2. Loading Whisper model: {MODEL_NAME}...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)

    # Prepare datasets
    print("\n3. Processing datasets...")
    dataset = dataset.map(
        lambda batch: prepare_data_for_training(batch, feature_extractor, tokenizer),
        remove_columns=dataset.column_names["train"],
        num_proc=NUM_WORKERS
    )

    # Load pre-trained model
    print("\n4. Loading pre-trained model...")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Metric
    metric = evaluate.load("wer")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
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
    print("\n5. Starting training...")
    print(f"   - Training samples: {len(dataset['train'])}")
    print(f"   - Eval samples: {len(dataset['test'])}")
    print(f"   - Epochs: {NUM_EPOCHS}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Learning rate: {LEARNING_RATE}")
    print("")

    trainer.train()

    # Save final model
    print("\n6. Saving model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print(f"\n✓ Training complete! Model saved to {OUTPUT_DIR}")
    print("\nYou can now use the model for transcription with:")
    print(f"  python transcribe.py --model {OUTPUT_DIR} --audio your_audio.mp3")


if __name__ == "__main__":
    main()
