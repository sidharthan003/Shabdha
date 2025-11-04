
# Shabdha

Shabdha is an end-to-end pipeline for converting courtroom transcripts into Indian Sign Language (ISL) or other target sign languages, with animation and delivery to screens or devices, and human-in-the-loop checks for legal safety.

---

## Project Flow

1. **Convert cleaned court transcripts**
  - Input: Cleaned text transcripts from court proceedings.
2. **Linguistic normalization**
  - Normalize and preprocess text for translation.
3. **ISL/Target-sign translation**
  - Translate normalized text into ISL or other sign language notation.
4. **Notation/Animation**
  - Convert sign notation into animation data (keypoints, skeletons).
  - Use MediaPipe and OpenCV to extract and process gesture keypoints.
5. **Avatar/Video/Subtitles Generation**
  - Render animated avatars or videos.
  - Generate subtitles (SRT) using Whisper for audio alignment.
6. **Delivery**
  - Output to courtroom screens or attendees’ devices.
7. **Human-in-loop checks**
  - Legal experts review and approve outputs for legal safety.

---

## Features
- Transcript cleaning and normalization
- Sign language translation (ISL/target)
- Keypoint extraction and animation (MediaPipe, OpenCV)
- Avatar/video/subtitle generation
- Human-in-the-loop review for legal compliance

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

### 1. Transcript Processing
- Place your cleaned transcript in the project directory.
- (Future) Use provided scripts for normalization and translation.

### 2. Keypoint Extraction & Animation
- Place gesture/sign videos in the project directory.
- Run extraction scripts:
  ```sh
  python CODES/gesture.py
  python CODES/hello_capture.py
  ```
- This generates CSVs with pose keypoints for each frame.

### 3. Labeling & Training Data Preparation
- Add gesture labels for supervised learning:
  ```sh
  python CODES/labels.py
  ```

### 4. Audio Transcription & Subtitle Generation
- Transcribe audio and generate SRT subtitles:
  ```sh
  python CODES/stage1_transcribe_segment.py
  ```

### 5. Avatar/Video/Subtitles Delivery
- Render avatars or videos from animation data (future work).
- Deliver outputs to screens or devices.

### 6. Human-in-the-loop Review
- Legal experts review outputs for compliance and safety.

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
- Future work: Integrate full pipeline for transcript-to-animation and delivery.

---

## License
MIT License

## Author
sidharthan003
