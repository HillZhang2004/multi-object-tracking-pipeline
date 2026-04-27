"""
CS153 Final Project: Multi-Object Tracking Pipeline
File: load_kitti_gt.py
Authors: Spencer Merodio, Hill Zhang
Date: 2026-04-26

AI Use:
We used an AI assistant in a limited way on this script:
  - Reformatting comments and the file header so the script is easier to read.
  - Checking that the parsing logic matched the KITTI tracking label format.
  - Sanity checking that the returned columns matched the evaluation script.

Summary:
Utility module for loading KITTI tracking ground truth annotations. The KITTI
tracking labels are stored as space-separated .txt files, one per sequence,
under:
    data/kitti/tracking/training/label_02/<sequence>.txt

Each row has the format:
    frame track_id class truncated occluded alpha x1 y1 x2 y2 h w l tx ty tz ry

This module reads those files and returns a tidy pandas DataFrame with the same
basic schema used by the SORT and Deep SORT track outputs, so the evaluation
script can compare predicted and ground-truth tracks.

Usage:
    from load_kitti_gt import load_gt_for_sequence, load_all_gt

    gt_df = load_gt_for_sequence("0000")
    all_gt = load_all_gt()
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path to KITTI tracking ground truth label files.
GT_DIR = PROJECT_ROOT / "data" / "kitti" / "tracking" / "training" / "label_02"

# KITTI class names that map to our tracked classes.
# We keep only the two main classes used in our project evaluation.
KITTI_CLASS_MAP = {
    "Car": "car",
    "Pedestrian": "person",
}

# KITTI label columns from the tracking benchmark format.
KITTI_COLS = [
    "frame",
    "track_id",
    "class_name",
    "truncated",
    "occluded",
    "alpha",
    "x1",
    "y1",
    "x2",
    "y2",
    "height",
    "width",
    "length",
    "tx",
    "ty",
    "tz",
    "ry",
]

GT_COLUMNS = [
    "sequence",
    "frame",
    "track_id",
    "class_name",
    "x1",
    "y1",
    "x2",
    "y2",
]


def load_gt_for_sequence(seq_name: str) -> pd.DataFrame:
    """
    Load ground truth annotations for one KITTI sequence.

    Parameters
    ----------
    seq_name : str
        Sequence identifier, for example "0000".

    Returns
    -------
    pd.DataFrame
        Columns: sequence, frame, track_id, class_name, x1, y1, x2, y2.
        Only Car and Pedestrian rows are kept and remapped to car/person.
        DontCare and other non-evaluated classes are dropped.
    """
    label_file = GT_DIR / f"{seq_name}.txt"

    if not label_file.exists():
        raise FileNotFoundError(f"GT label file not found: {label_file}")

    df = pd.read_csv(label_file, sep=r"\s+", header=None, names=KITTI_COLS)

    # Keep only the classes used in our project evaluation.
    df = df[df["class_name"].isin(KITTI_CLASS_MAP)].copy()
    df["class_name"] = df["class_name"].map(KITTI_CLASS_MAP)
    df["sequence"] = seq_name

    df["track_id"] = df["track_id"].astype(int)
    df["frame"] = df["frame"].astype(int)

    return df[GT_COLUMNS]


def load_all_gt() -> pd.DataFrame:
    """
    Load ground truth annotations for all sequences found in GT_DIR.

    Returns
    -------
    pd.DataFrame
        Same schema as load_gt_for_sequence, concatenated across all sequences.
    """
    if not GT_DIR.exists():
        raise FileNotFoundError(f"GT directory not found: {GT_DIR}")

    all_dfs = []

    for label_file in sorted(GT_DIR.glob("*.txt")):
        seq_name = label_file.stem
        try:
            all_dfs.append(load_gt_for_sequence(seq_name))
        except Exception as error:
            print(f"Warning: could not load GT for {seq_name}: {error}")

    if not all_dfs:
        raise RuntimeError(f"No ground truth label files found in {GT_DIR}")

    return pd.concat(all_dfs, ignore_index=True)