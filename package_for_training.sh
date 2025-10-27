#!/bin/bash
# Package everything needed for training on RTX 3090 machine

echo "=================================="
echo "Packaging Georgian STT for Transfer"
echo "=================================="

# Create a temp directory
mkdir -p package_transfer

# Copy Python scripts
echo "Copying scripts..."
cp 2_finetune_whisper.py package_transfer/
cp 3_transcribe.py package_transfer/
cp 4_gradio_app.py package_transfer/
cp requirements.txt package_transfer/
cp SETUP_RTX3090.md package_transfer/README.md

# Create archive of dataset (this will take a while - 6.9GB file)
echo ""
echo "⏳ Creating dataset archive (this may take 5-10 minutes)..."
tar -czf package_transfer/georgian_dataset.tar.gz cv-corpus-23.0-2025-09-05/

echo ""
echo "✓ Done! Package created in: package_transfer/"
echo ""
echo "Transfer to your RTX 3090 machine with:"
echo "  scp -r package_transfer/ user@your-machine:/path/to/training/"
echo ""
echo "Or upload to cloud storage and download on training machine."
echo ""
echo "Contents:"
ls -lh package_transfer/
