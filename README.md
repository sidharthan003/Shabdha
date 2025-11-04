
# Shabdha

Shabdha is a Python-based project for gesture recognition and keypoint extraction from videos, featuring pose analysis, gesture labeling, and audio transcription. It is designed for building gesture datasets and training machine learning models for sign/gesture recognition.

---

## Features
- Extracts pose keypoints from gesture videos using MediaPipe Holistic
- Flattens and saves keypoints to CSV files for each frame
- Adds gesture labels for supervised learning
- Transcribes audio to subtitles using OpenAI Whisper
- Example scripts for each step of the workflow

---

## Requirements
- Python 3.8+
- OpenCV
- MediaPipe
- pandas
- numpy
- torch (for Whisper)
- openai-whisper
- nltk
- pysrt

---

## Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/sidharthan003/Shabdha.git
   cd Shabdha
   ```
2. (Recommended) Create and activate a virtual environment:
   ```sh
   python -m venv shabdha
   # On Windows:
   shabdha\Scripts\activate
   # On Unix/Mac:
   source shabdha/bin/activate
   ```
3. Install dependencies:
   ```sh
   pip install opencv-python mediapipe pandas numpy torch openai-whisper nltk pysrt
   ```

---

## Workflow & Usage

### 1. Extract Keypoints from Video

- Place your gesture video (e.g., `sample_gesture.mp4` or `hello_gesture.mp4`) in the project directory.
- Run the extraction script:
  ```sh
  python CODES/gesture.py
  # or for hello gesture
  python CODES/hello_capture.py
  ```
- This will generate a CSV file (e.g., `gesture_keypoints.csv` or `hello_keypoints.csv`) with pose keypoints for each frame.

### 2. Label Keypoints for Training

- Add a label to each frame or sequence for supervised learning:
  ```sh
  python CODES/labels.py
  ```
- This will create a labeled CSV (e.g., `gesture_keypoints_labeled.csv`).

### 3. Transcribe Audio to Subtitles

- To align gestures with speech, transcribe audio using Whisper:
  ```sh
  python CODES/stage1_transcribe_segment.py
  ```
- This will generate an SRT subtitle file (e.g., `output_transcript.srt`).

### 4. (Optional) Test Whisper Model & CUDA

- Check CUDA and Whisper setup:
  ```sh
  python CODES/whisper_test.py
  ```

---

## Example Scripts
- `CODES/gesture.py`: Extracts pose keypoints from a video and saves to CSV.
- `CODES/hello_capture.py`: Specialized for "hello" gesture video.
- `CODES/labels.py`: Adds a label column to a keypoints CSV.
- `CODES/stage1_transcribe_segment.py`: Transcribes audio and creates SRT subtitles.
- `CODES/whisper_test.py`: Checks CUDA and Whisper model setup.

---

## Notes
- Add all generated CSV and SRT files to `.gitignore` to avoid tracking large data files.
- For best results, use clear, well-lit videos with visible gestures.
- You can segment gestures using subtitles or manual annotation if your video contains multiple gestures.
- Use the labeled CSVs to train gesture recognition models (e.g., LSTM, CNN, Transformer).

---

## License
MIT License

## Author
sidharthan003
