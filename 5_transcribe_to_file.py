"""
Enhanced transcription script that saves results to text files
Supports single files, multiple files, and batch processing
"""

import argparse
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
from pathlib import Path
from datetime import datetime
import os


def format_duration(seconds):
    """Format duration in seconds to readable string"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes > 0:
        return f"{minutes} minute{'s' if minutes != 1 else ''} {secs} second{'s' if secs != 1 else ''}"
    else:
        return f"{secs} second{'s' if secs != 1 else ''}"


def transcribe_audio(audio_path, model, processor, language="ka", device="cpu"):
    """
    Transcribe a single audio file

    Args:
        audio_path: Path to audio file
        model: Loaded Whisper model
        processor: Loaded Whisper processor
        language: Language code (default: ka for Georgian)
        device: Device to run on (cuda/cpu)

    Returns:
        tuple: (transcription_text, audio_duration)
    """
    # Load audio
    print(f"  Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr

    # Process audio
    input_features = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features

    # Move to device
    input_features = input_features.to(device)

    # Generate transcription
    print(f"  Generating transcription...")
    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    # Decode transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription, duration


def save_transcription(audio_path, transcription, duration, model_name, output_path=None):
    """
    Save transcription to a formatted text file

    Args:
        audio_path: Original audio file path
        transcription: Transcribed text
        duration: Audio duration in seconds
        model_name: Name of the model used
        output_path: Optional custom output path

    Returns:
        Path to saved text file
    """
    # Determine output path
    if output_path is None:
        audio_file = Path(audio_path)
        output_path = audio_file.parent / f"{audio_file.stem}_transcription.txt"
    else:
        output_path = Path(output_path)

    # Create formatted content
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration_str = format_duration(duration)

    content = f"""Georgian Speech Transcription
{'=' * 60}

Generated: {timestamp}
Model: {model_name}
Audio File: {Path(audio_path).name}
Duration: {duration_str}

{'─' * 60}

{transcription}

{'─' * 60}

Transcription completed successfully.
"""

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return output_path


def process_single_file(audio_path, model, processor, model_name, output_path=None, language="ka", device="cpu"):
    """Process a single audio file and save transcription"""
    print(f"\nProcessing: {audio_path}")
    print("-" * 60)

    # Transcribe
    transcription, duration = transcribe_audio(audio_path, model, processor, language, device)

    # Save to file
    output_file = save_transcription(audio_path, transcription, duration, model_name, output_path)

    print(f"\n  Transcription:")
    print(f"  {transcription}")
    print(f"\n  Saved to: {output_file}")
    print("-" * 60)

    return output_file


def process_folder(folder_path, model, processor, model_name, output_folder=None, language="ka", device="cpu"):
    """Process all audio files in a folder"""
    folder = Path(folder_path)

    # Supported audio formats
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.webm']

    # Find all audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(folder.glob(f"*{ext}"))

    if not audio_files:
        print(f"No audio files found in {folder_path}")
        return []

    print(f"\nFound {len(audio_files)} audio file(s) to process")
    print("=" * 60)

    # Create output folder if specified
    if output_folder:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

    # Process each file
    output_files = []
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        print("-" * 60)

        try:
            # Determine output path
            if output_folder:
                output_path = output_folder / f"{audio_file.stem}_transcription.txt"
            else:
                output_path = None

            # Transcribe
            transcription, duration = transcribe_audio(audio_file, model, processor, language, device)

            # Save
            output_file = save_transcription(audio_file, transcription, duration, model_name, output_path)
            output_files.append(output_file)

            print(f"  Transcription preview: {transcription[:100]}{'...' if len(transcription) > 100 else ''}")
            print(f"  Saved to: {output_file}")
            print(f"  ✓ Completed")

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            continue

    print("\n" + "=" * 60)
    print(f"Batch processing complete!")
    print(f"Successfully processed: {len(output_files)}/{len(audio_files)} files")
    print("=" * 60)

    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files and save to text files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe single file
  python 5_transcribe_to_file.py --audio my_audio.mp3 --model ./whisper-georgian-local

  # Transcribe with custom output location
  python 5_transcribe_to_file.py --audio my_audio.mp3 --model ./whisper-georgian-local --output result.txt

  # Batch process folder
  python 5_transcribe_to_file.py --audio-folder ./my_podcasts/ --model ./whisper-georgian-local

  # Batch process with output folder
  python 5_transcribe_to_file.py --audio-folder ./podcasts/ --model ./whisper-georgian-local --output-folder ./transcriptions/
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio", type=str, help="Path to single audio file")
    input_group.add_argument("--audio-folder", type=str, help="Path to folder containing audio files")

    # Model options
    parser.add_argument("--model", type=str, default="./whisper-georgian-local",
                        help="Path to fine-tuned model (default: ./whisper-georgian-local)")
    parser.add_argument("--language", type=str, default="ka",
                        help="Language code (default: ka for Georgian)")

    # Output options
    parser.add_argument("--output", type=str,
                        help="Output file path (for single file mode)")
    parser.add_argument("--output-folder", type=str,
                        help="Output folder path (for folder mode)")

    # Device options
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto",
                        help="Device to use (default: auto)")

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Print header
    print("=" * 60)
    print("GEORGIAN SPEECH-TO-TEXT TRANSCRIPTION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {device.upper()}")
    if torch.cuda.is_available() and device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # Load model and processor
    print("\nLoading model...")
    try:
        processor = WhisperProcessor.from_pretrained(args.model)
        model = WhisperForConditionalGeneration.from_pretrained(args.model)
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        print(f"\nMake sure the model exists at: {args.model}")
        print("If you haven't trained a model yet, run: python 2_finetune_whisper_local.py")
        return

    # Process audio
    try:
        if args.audio:
            # Single file mode
            if not os.path.exists(args.audio):
                print(f"Error: Audio file not found: {args.audio}")
                return

            output_file = process_single_file(
                args.audio, model, processor, args.model,
                args.output, args.language, device
            )

            print(f"\n✓ Done! Transcription saved to: {output_file}")

        else:
            # Folder mode
            if not os.path.exists(args.audio_folder):
                print(f"Error: Folder not found: {args.audio_folder}")
                return

            output_files = process_folder(
                args.audio_folder, model, processor, args.model,
                args.output_folder, args.language, device
            )

            if output_files:
                print(f"\n✓ Done! Transcriptions saved to:")
                for f in output_files[:5]:  # Show first 5
                    print(f"  - {f}")
                if len(output_files) > 5:
                    print(f"  ... and {len(output_files) - 5} more")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n✗ Error during transcription: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
