import mediapipe as mp
import cv2
import pandas as pd
import numpy as np

# Set up MediaPipe Holistic
mp_holistic = mp.solutions.holistic

# Open the video file
cap = cv2.VideoCapture('hello_gesture.mp4')
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit(1)

keypoints = []
with mp_holistic.Holistic(static_image_mode=False) as holistic:
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            # Flatten all keypoints for this frame
            pose = []
            for l in results.pose_landmarks.landmark:
                pose.extend([l.x, l.y, l.z])
            keypoints.append(pose)
        else:
            # If no pose detected, append NaNs
            keypoints.append([np.nan] * 99)  # 33 landmarks * 3 coords
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

cap.release()

# Save to CSV
df = pd.DataFrame(keypoints)
df.to_csv('hello_keypoints.csv', index=False)
print("Keypoints saved to hello_keypoints.csv")