import pandas as pd

df = pd.read_csv("gesture_keypoints.csv")
df["label"] = "hello"  # or any gesture name for this recording
df.to_csv("gesture_keypoints_labeled.csv", index=False)
