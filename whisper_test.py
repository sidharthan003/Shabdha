import whisper
import torch

# Check if GPU is available
print("CUDA available:", torch.cuda.is_available())

# Load a small Whisper model on GPU
model = whisper.load_model("tiny", device="cuda")  # tiny model for fast test

# Print model info
print("Model loaded on device:", model.device)
