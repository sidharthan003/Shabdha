import whisper
import torch
import sounddevice as sd
import numpy as np

# Load tiny model on GPU for fast test
model = whisper.load_model("tiny", device="cuda")

# Audio recording settings
duration = 5  # seconds
sample_rate = 16000

print("Recording for", duration, "seconds...")

# Record audio from mic
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()
audio = np.squeeze(audio)

# Whisper expects float32 in range [-1, 1]
result = model.transcribe(audio, language="en")

print("Transcribed text:")
print(result["text"])
