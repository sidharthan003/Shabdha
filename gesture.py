import mediapipe as mp
import cv2
import pandas as pd

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture('sample_gesture.mp4')
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit(1)

with mp_holistic.Holistic(static_image_mode=False) as holistic:
    keypoints = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"End of video or failed to read frame at index {frame_count}.")
            break
        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            pose = [[l.x, l.y, l.z] for l in results.pose_landmarks.landmark]
            keypoints.append(pose)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")
    cap.release()

pd.DataFrame(keypoints).to_csv('gesture_keypoints.csv', index=False)
