"""
Transcribe audio files using fine-tuned Whisper model
"""

import argparse
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa


def transcribe_audio(audio_path, model, processor, language="ka"):
    """Transcribe a single audio file"""

    # Load audio
    print(f"Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000)

    # Process audio
    input_features = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_features = input_features.to(device)
    model = model.to(device)

    # Generate transcription
    print("Generating transcription...")
    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    # Decode transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--model", type=str, default="./whisper-georgian-finetuned",
                        help="Path to fine-tuned model")
    parser.add_argument("--language", type=str, default="ka", help="Language code (default: ka for Georgian)")

    args = parser.parse_args()

    # Load model and processor
    print(f"Loading model from {args.model}...")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)

    # Transcribe
    transcription = transcribe_audio(args.audio, model, processor, args.language)

    print("\n" + "=" * 50)
    print("TRANSCRIPTION:")
    print("=" * 50)
    print(transcription)
    print("=" * 50)

    return transcription


if __name__ == "__main__":
    main()
