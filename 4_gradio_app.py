"""
Simple Gradio web interface for Georgian speech transcription
Upload audio and get transcription
"""

import gradio as gr
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import numpy as np


class GeorgianTranscriber:
    def __init__(self, model_path="./whisper-georgian-finetuned"):
        """Initialize the transcriber with a fine-tuned model"""
        print(f"Loading model from {model_path}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        print("Model loaded successfully!")

    def transcribe(self, audio_input):
        """
        Transcribe audio input
        Args:
            audio_input: tuple of (sample_rate, audio_data) from Gradio
        Returns:
            transcription text
        """
        if audio_input is None:
            return "Please upload or record audio."

        try:
            # Gradio returns (sample_rate, audio_array)
            sample_rate, audio_data = audio_input

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

            return transcription

        except Exception as e:
            return f"Error during transcription: {str(e)}"


def create_interface(model_path="./whisper-georgian-finetuned"):
    """Create Gradio interface"""

    # Initialize transcriber
    transcriber = GeorgianTranscriber(model_path)

    # Create interface
    interface = gr.Interface(
        fn=transcriber.transcribe,
        inputs=gr.Audio(sources=["upload", "microphone"], type="numpy"),
        outputs=gr.Textbox(label="Transcription (Georgian)", lines=5),
        title="🎙️ Georgian Speech-to-Text",
        description="""
        Upload an audio file or record your voice to get Georgian transcription.

        This model is fine-tuned Whisper for Georgian language.
        """,
        examples=[],
        theme=gr.themes.Soft(),
    )

    return interface


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./whisper-georgian-finetuned",
                        help="Path to fine-tuned model")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")

    args = parser.parse_args()

    # Create and launch interface
    interface = create_interface(args.model)
    interface.launch(
        share=args.share,
        server_port=args.port
    )


if __name__ == "__main__":
    main()
