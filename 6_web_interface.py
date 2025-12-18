"""
Enhanced Gradio web interface for Georgian speech transcription
Features: Upload audio, view transcription, download text file
"""

import gradio as gr
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile
import os


def format_duration(seconds):
    """Format duration in seconds to readable string"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class GeorgianTranscriber:
    def __init__(self, model_path="./whisper-georgian-local"):
        """Initialize the transcriber with a fine-tuned model"""
        print(f"Loading model from {model_path}...")
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")

    def transcribe(self, audio_input):
        """
        Transcribe audio input and return both text and downloadable file

        Args:
            audio_input: Either:
                - tuple of (sample_rate, audio_data) from Gradio microphone
                - str path to uploaded file

        Returns:
            tuple: (transcription_text, info_text, download_file_path)
        """
        if audio_input is None:
            return "Please upload or record audio.", "No audio provided.", None

        try:
            # Handle different input types
            if isinstance(audio_input, str):
                # File path from upload
                audio_path = audio_input
                audio_data, sample_rate = librosa.load(audio_path, sr=16000)
                original_filename = Path(audio_path).stem
            else:
                # Tuple from microphone
                sample_rate, audio_data = audio_input
                original_filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                # Convert to float and normalize if needed
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0

                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    audio_data = librosa.resample(
                        audio_data,
                        orig_sr=sample_rate,
                        target_sr=16000
                    )

            # Calculate duration
            duration = len(audio_data) / 16000

            # Process audio
            input_features = self.processor(
                audio_data,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features

            input_features = input_features.to(self.device)

            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)

            # Decode
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            # Create info text
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            duration_str = format_duration(duration)
            info = f"✓ Transcribed successfully\nDuration: {duration_str} | Time: {timestamp}"

            # Create downloadable text file
            download_file = self._create_text_file(
                transcription, original_filename, duration, timestamp
            )

            return transcription, info, download_file

        except Exception as e:
            error_msg = f"Error during transcription: {str(e)}"
            return error_msg, "✗ Transcription failed", None

    def _create_text_file(self, transcription, filename, duration, timestamp):
        """Create a formatted text file for download"""
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"{filename}_transcription.txt")

        duration_str = format_duration(duration)

        content = f"""Georgian Speech Transcription
{'=' * 60}

Generated: {timestamp}
Model: {Path(self.model_path).name}
Audio File: {filename}
Duration: {duration_str}

{'─' * 60}

{transcription}

{'─' * 60}

Transcription completed successfully.
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return output_path


def create_interface(model_path="./whisper-georgian-local"):
    """Create enhanced Gradio interface with download capability"""

    # Initialize transcriber
    transcriber = GeorgianTranscriber(model_path)

    # Custom CSS for better styling
    custom_css = """
    .output-text {
        font-size: 18px !important;
        line-height: 1.6 !important;
    }
    .info-text {
        font-size: 14px !important;
        color: #666 !important;
    }
    .header {
        text-align: center;
        margin-bottom: 20px;
    }
    """

    # Create interface
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # Georgian Speech-to-Text Transcription

            Upload an audio file or record your voice to get Georgian transcription.

            **Supported formats:** MP3, WAV, M4A, FLAC, OGG
            """,
            elem_classes=["header"]
        )

        with gr.Row():
            with gr.Column():
                # Audio input
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Audio Input"
                )

                # Submit button
                submit_btn = gr.Button("Transcribe", variant="primary", size="lg")

            with gr.Column():
                # Transcription output
                transcription_output = gr.Textbox(
                    label="Transcription (Georgian)",
                    lines=10,
                    placeholder="Transcription will appear here...",
                    elem_classes=["output-text"]
                )

                # Info output
                info_output = gr.Textbox(
                    label="Status",
                    lines=2,
                    elem_classes=["info-text"]
                )

                # Download button
                download_output = gr.File(
                    label="Download Transcription (.txt)",
                    interactive=False
                )

        # Instructions
        gr.Markdown(
            """
            ---
            ### How to use:

            1. **Upload audio** or **record** your voice using the input above
            2. Click **Transcribe** button
            3. View the transcription in the text box
            4. Click the **Download** button to save as a text file

            ### Tips:
            - Best results with clear audio and minimal background noise
            - Works with Georgian language audio
            - Longer files may take more time to process
            """
        )

        # Connect the button to the function
        submit_btn.click(
            fn=transcriber.transcribe,
            inputs=[audio_input],
            outputs=[transcription_output, info_output, download_output]
        )

        # Example section (optional - can add example files later)
        gr.Markdown(
            """
            ---
            <div style='text-align: center; color: #666; font-size: 12px;'>
            Powered by fine-tuned Whisper model for Georgian language
            </div>
            """
        )

    return interface


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch web interface for Georgian speech-to-text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch with default model
  python 6_web_interface.py

  # Launch with custom model
  python 6_web_interface.py --model ./my_model/

  # Create public link (share with others)
  python 6_web_interface.py --share

  # Use custom port
  python 6_web_interface.py --port 8080
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default="./whisper-georgian-local",
        help="Path to fine-tuned model (default: ./whisper-georgian-local)"
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public shareable link (accessible from internet)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on (default: 7860)"
    )

    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Server name/IP (default: 127.0.0.1 for local only, use 0.0.0.0 for network access)"
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("\nPlease train a model first by running:")
        print("  python 2_finetune_whisper_local.py")
        print("\nOr specify a different model path with --model")
        return

    print("=" * 70)
    print("GEORGIAN SPEECH-TO-TEXT WEB INTERFACE")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Port: {args.port}")
    print(f"Share: {'Yes (public link)' if args.share else 'No (local only)'}")
    print("=" * 70)

    # Create and launch interface
    try:
        interface = create_interface(args.model)

        print("\nStarting server...")
        print(f"\nLocal URL: http://{args.server_name}:{args.port}")

        if args.share:
            print("\nGenerating public share link...")
            print("(This link will be accessible from anywhere on the internet)")

        print("\n" + "=" * 70)
        print("Server is running! Press Ctrl+C to stop.")
        print("=" * 70)

        interface.launch(
            share=args.share,
            server_port=args.port,
            server_name=args.server_name,
            show_error=True
        )

    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")
    except Exception as e:
        print(f"\nError starting server: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
