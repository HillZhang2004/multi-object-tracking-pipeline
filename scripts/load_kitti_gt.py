"""
CS153 Final Project: Multi-Object Tracking Pipeline
File: load_kitti_gt.py
Authors: Spencer Merodio, Hill Zhang
Date: 2026-04-07

AI Use:
We used an AI assistant in a limited way on this script:
  - Reformatting comments and the file header so the script is easier to read.
  - Checking that the parsing logic matched the KITTI label file format spec.

Summary:
Utility module for loading KITTI tracking ground truth annotations. The KITTI
tracking labels are stored as space-separated .txt files, one per sequence,
under:
    data/kitti/tracking/training/label_02/<sequence>.txt

Each row has the format:
    frame track_id class truncated occluded alpha x1 y1 x2 y2 h w l tx ty tz ry

This module reads those files and returns a tidy pandas DataFrame in the same
schema used by the SORT and Deep SORT track outputs, so the evaluation script
can compare predicted and ground-truth tracks directly.

Usage:
    from load_kitti_gt import load_gt_for_sequence, load_all_gt

    # Load one sequence
    gt_df = load_gt_for_sequence("0000")

    # Load all sequences
    all_gt = load_all_gt()
"""

from pathlib import Path
import pandas as pd

# Path to KITTI tracking ground truth label files
GT_DIR = Path("data/kitti/tracking/training/label_02")

# KITTI class names that map to our tracked classes.
# KITTI uses "Car", "Pedestrian"; our pipeline uses lowercase "car", "person".
KITTI_CLASS_MAP = {
    "Car": "car",
    "Pedestrian": "person",
    "Van": "car",       # optional: fold Van into car for evaluation
    "Person_sitting": "person",
}

# KITTI label columns (from the tracking benchmark spec)
KITTI_COLS = [
    "frame", "track_id", "class_name",
    "truncated", "occluded", "alpha",
    "x1", "y1", "x2", "y2",
    "height", "width", "length",
    "tx", "ty", "tz", "ry",
]


def load_gt_for_sequence(seq_name: str) -> pd.DataFrame:
    """
    Load ground truth annotations for a single KITTI sequence.

    Parameters
    ----------
    seq_name : str
        Sequence identifier, e.g. "0000".

    Returns
    -------
    pd.DataFrame
        Columns: sequence, frame, track_id, class_name, x1, y1, x2, y2
        Only rows whose class_name is in KITTI_CLASS_MAP are returned.
        The "DontCare" class and other irrelevant classes are dropped.
    """
    label_file = GT_DIR / f"{seq_name}.txt"
    if not label_file.exists():
        raise FileNotFoundError(f"GT label file not found: {label_file}")

    df = pd.read_csv(label_file, sep=" ", header=None, names=KITTI_COLS)

    # Keep only the classes we care about and remap to our lowercase names.
    df = df[df["class_name"].isin(KITTI_CLASS_MAP)].copy()
    df["class_name"] = df["class_name"].map(KITTI_CLASS_MAP)
    df["sequence"] = seq_name

    # KITTI uses -1 for track_id on DontCare rows; those are already filtered
    # but cast to int cleanly anyway.
    df["track_id"] = df["track_id"].astype(int)
    df["frame"] = df["frame"].astype(int)

    return df[["sequence", "frame", "track_id", "class_name", "x1", "y1", "x2", "y2"]]


def load_all_gt() -> pd.DataFrame:
    """
    Load ground truth annotations for all sequences found in GT_DIR.

    Returns
    -------
    pd.DataFrame
        Same schema as load_gt_for_sequence, concatenated across all sequences.
    """
    all_dfs = []
    for label_file in sorted(GT_DIR.glob("*.txt")):
        seq_name = label_file.stem
        try:
            all_dfs.append(load_gt_for_sequence(seq_name))
        except Exception as e:
            print(f"Warning: could not load GT for {seq_name}: {e}")

    if not all_dfs:
        raise RuntimeError(f"No ground truth label files found in {GT_DIR}")

    return pd.concat(all_dfs, ignore_index=True)