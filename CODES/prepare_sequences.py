import argparse
import ast
import numpy as np
import pandas as pd


def build_sequences(data, seq_len, stride):
    sequences = []
    for start in range(0, len(data) - seq_len + 1, stride):
        sequences.append(data[start:start + seq_len])
    if not sequences:
        return np.empty((0, seq_len, data.shape[1]), dtype=data.dtype)
    return np.stack(sequences, axis=0)


def load_pose_csv(df):
    if df.shape[1] == 99:
        return df.values

    if df.shape[1] == 33:
        frames = []
        for _, row in df.iterrows():
            pose = []
            for cell in row.values:
                if isinstance(cell, str):
                    coords = ast.literal_eval(cell)
                else:
                    coords = cell
                pose.extend(coords)
            frames.append(pose)
        return np.array(frames, dtype=float)

    raise SystemExit("Unsupported CSV format. Expected 33 or 99 columns.")


def main():
    parser = argparse.ArgumentParser(description="Prepare fixed-length keypoint sequences from a CSV.")
    parser.add_argument("--input", required=True, help="Input CSV path with keypoints.")
    parser.add_argument("--output", required=True, help="Output NPZ file path.")
    parser.add_argument("--seq-len", type=int, default=30, help="Sequence length in frames.")
    parser.add_argument("--stride", type=int, default=10, help="Stride between sequences.")
    parser.add_argument("--label-col", default=None, help="Optional label column name.")
    parser.add_argument("--label", default=None, help="Optional constant label to assign to all sequences.")
    parser.add_argument("--drop-na", action="store_true", help="Drop rows with any NaN values.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.drop_na:
        df = df.dropna()

    labels = None
    if args.label_col and args.label_col in df.columns:
        labels = df[args.label_col].copy()
        df = df.drop(columns=[args.label_col])
    elif args.label is not None:
        labels = pd.Series([args.label] * len(df))

    data = load_pose_csv(df)
    if data.size == 0:
        raise SystemExit("No data found in the input CSV.")

    X = build_sequences(data, args.seq_len, args.stride)
    if X.shape[0] == 0:
        raise SystemExit("Not enough frames to build sequences with the given length.")

    if labels is not None:
        y_seq = []
        for start in range(0, len(labels) - args.seq_len + 1, args.stride):
            window = labels.iloc[start:start + args.seq_len]
            if window.isna().all():
                y_seq.append(-1)
            else:
                y_seq.append(window.mode().iloc[0])
        y = np.array(y_seq)
        np.savez(args.output, X=X, y=y)
    else:
        np.savez(args.output, X=X)

    print(f"Saved sequences to {args.output}")


if __name__ == "__main__":
    main()
