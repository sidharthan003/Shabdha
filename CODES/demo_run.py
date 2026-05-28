import argparse
import numpy as np
import pandas as pd
from prepare_sequences import build_sequences, load_pose_csv
from visualize_skeleton import load_frame


def main():
    parser = argparse.ArgumentParser(description="Quick demo: build sequences and save a frame plot.")
    parser.add_argument("--input", default="CSV/gesture_keypoints.csv", help="Input CSV path.")
    parser.add_argument("--seq-len", type=int, default=30, help="Sequence length in frames.")
    parser.add_argument("--stride", type=int, default=10, help="Stride between sequences.")
    parser.add_argument("--plot", default="CSV/pose_demo.png", help="Output plot path.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    data = load_pose_csv(df)
    X = build_sequences(data, args.seq_len, args.stride)
    if X.shape[0] == 0:
        raise SystemExit("Not enough frames to build sequences.")
    np.savez("CSV/sequences_demo.npz", X=X)
    print("Saved CSV/sequences_demo.npz")

    pose = load_frame(args.input, 0)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import mediapipe as mp

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], s=10)

    for a, b in mp.solutions.holistic.POSE_CONNECTIONS:
        ax.plot([pose[a, 0], pose[b, 0]],
                [pose[a, 1], pose[b, 1]],
                [pose[a, 2], pose[b, 2]],
                "r-", linewidth=1)

    ax.invert_yaxis()
    plt.savefig(args.plot, dpi=150, bbox_inches="tight")
    print(f"Saved {args.plot}")


if __name__ == "__main__":
    main()
