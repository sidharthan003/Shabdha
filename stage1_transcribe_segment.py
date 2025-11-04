import whisper
import nltk
import pysrt

# Download necessary models
nltk.download('punkt')
nltk.download('punkt_tab')

# Load Whisper model on GPU
model = whisper.load_model("medium", device="cuda")

# Input audio file
audio_file = "courtroom_sample.mp3"

print(f"Transcribing {audio_file} ...")
result = model.transcribe(audio_file, language="en")

segments = result["segments"]  # contains text + timestamps
print("\n--- SEGMENTED TRANSCRIPT ---")

subs = pysrt.SubRipFile()

for i, seg in enumerate(segments, start=1):
    start = seg["start"]
    end = seg["end"]
    text = seg["text"].strip()

    # Display on console
    print(f"[{start:.2f} - {end:.2f}] {text}")

    # Add to SRT
    subs.append(
        pysrt.SubRipItem(
            index=i,
            start=pysrt.SubRipTime(seconds=start),
            end=pysrt.SubRipTime(seconds=end),
            text=text
        )
    )

# Save as SRT file
subs.save("output_transcript.srt", encoding="utf-8")

print("\n✅ SRT file saved as 'output_transcript.srt'")
