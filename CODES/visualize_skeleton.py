import argparse
import ast
import numpy as np
import pandas as pd


def load_frame(csv_path, frame_index, landmarks=33):
    df = pd.read_csv(csv_path)
    if frame_index < 0 or frame_index >= len(df):
        raise SystemExit("Frame index out of range.")
    row = df.iloc[frame_index].values

    if row.size == landmarks * 3:
        return row.reshape(landmarks, 3)

    if row.size == landmarks:
        pose = []
        for cell in row:
            if isinstance(cell, str):
                coords = ast.literal_eval(cell)
            else:
                coords = cell
            pose.append(coords)
        return np.array(pose, dtype=float)

    raise SystemExit("Unsupported CSV format. Expected 33 or 99 columns.")


def main():
    parser = argparse.ArgumentParser(description="Render a 3D pose skeleton from a keypoints CSV.")
    parser.add_argument("--input", required=True, help="Input CSV path with pose keypoints.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to render.")
    parser.add_argument("--save", default=None, help="Optional output image path.")
    args = parser.parse_args()

    if args.save:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import mediapipe as mp

    pose = load_frame(args.input, args.frame)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], s=10)

    for a, b in mp.solutions.holistic.POSE_CONNECTIONS:
        ax.plot([pose[a, 0], pose[b, 0]],
                [pose[a, 1], pose[b, 1]],
                [pose[a, 2], pose[b, 2]],
                "r-", linewidth=1)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.invert_yaxis()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
