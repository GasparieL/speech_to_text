"""
Quick test of pre-trained Whisper on Georgian audio
Run this FIRST before fine-tuning to see baseline performance
"""

import whisper
import sys
from pathlib import Path

# Ensure UTF-8 output for Georgian text
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("="*60)
print("Testing Pre-trained Whisper on Georgian Speech")
print("="*60)

# Configuration
MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large
AUDIO_DIR = Path("cv-corpus-23.0-2025-09-05/ka/clips")

# Find first audio file
audio_files = list(AUDIO_DIR.glob("*.mp3"))
if not audio_files:
    print("ERROR: Error: No audio files found!")
    print(f"   Looking in: {AUDIO_DIR}")
    sys.exit(1)

test_audio = audio_files[0]
print(f"\nOK: Found {len(audio_files)} audio files")
print(f"OK: Testing with: {test_audio.name}")

# Load model
print(f"\nLoading: Loading Whisper {MODEL_SIZE} model...")
model = whisper.load_model(MODEL_SIZE)
print("OK: Model loaded!")

# Transcribe
print("\nLoading: Transcribing (this may take 10-30 seconds)...")
result = model.transcribe(
    str(test_audio),
    language="ka",  # Georgian
    verbose=False
)

# Display results
print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(f"Audio: {test_audio.name}")
print(f"\nTranscription:")
print(f"  {result['text']}")
print("="*60)

# Load ground truth from TSV
train_tsv = Path("cv-corpus-23.0-2025-09-05/ka/train.tsv")
if train_tsv.exists():
    import pandas as pd
    df = pd.read_csv(train_tsv, sep='\t')
    ground_truth_row = df[df['path'] == test_audio.name]

    if not ground_truth_row.empty:
        ground_truth = ground_truth_row.iloc[0]['sentence']
        print(f"\nGround Truth:")
        print(f"  {ground_truth}")
        print("="*60)

        # Simple accuracy check
        if result['text'].strip().lower() == ground_truth.strip().lower():
            print("\nOK: Perfect match!")
        else:
            print("\nWARNING: Not an exact match (this is normal)")
            print("  Fine-tuning will improve accuracy significantly!")

# Test a few more files
print("\n\nTesting 3 more samples...")
for i, audio_file in enumerate(audio_files[1:4], 1):
    print(f"\n{i}. {audio_file.name}")
    result = model.transcribe(str(audio_file), language="ka", verbose=False)
    print(f"   ->{result['text']}")

print("\n" + "="*60)
print("OK: Test complete!")
print("\nNext steps:")
print("1. If the transcriptions look reasonable ->Great!")
print("2. To improve accuracy ->Fine-tune on your data")
print("3. On Mac ->Use Google Colab (see COLAB_TRAINING.ipynb)")
print("4. With GPU ->Run: python 2_finetune_whisper.py")
print("="*60)
