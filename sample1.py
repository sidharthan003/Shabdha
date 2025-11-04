import whisper
import torch

# Load Whisper model on GPU
model = whisper.load_model("tiny", device="cuda")  # tiny model is fast, good for testing

# Path to your audio file
audio_file = "sampleaudio.aac"  # replace with your file

# Transcribe the audio
result = model.transcribe(audio_file, language="en")  # language="ml" for Malayalam

# Print the transcription
print("Transcribed text:")
print(result["text"])
