"""
CS153 Final Project: Multi-Object Tracking Pipeline
File: visualize.py
Authors: Spencer Merodio, Hill Zhang
Date: 2026-04-07

AI Use:
We used an AI assistant in a limited way on this script:
  - Reformatting comments and the file header so the script is easier to read.
  - Checking that the code structure matched the project plan from our proposal.
  - Sanity checking the side-by-side frame layout and OpenCV video writer setup.

Summary:
This script renders tracking results as video. It supports three modes:

  --mode sort        Single video showing SORT tracks only
  --mode deepsort    Single video showing Deep SORT tracks only
  --mode compare     Side-by-side video (SORT on left, Deep SORT on right)
                     This is the main deliverable for comparing the two methods.

For each frame, bounding boxes are drawn with a unique color per track ID.
The class name and track ID are labeled above each box. In compare mode, the
two panels are placed side-by-side with a title bar labeling each side.

Ground truth boxes can optionally be overlaid in white using --show-gt.

Output videos are saved to:
    outputs/visualization/<seq_name>_<mode>.mp4

Usage:
    python scripts/visualize.py --mode compare
    python scripts/visualize.py --mode sort
    python scripts/visualize.py --mode deepsort --show-gt
"""

import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from load_kitti_gt import load_gt_for_sequence

# ─── Paths ────────────────────────────────────────────────────────────────────

SORT_DIR = Path("outputs/sort_tracks")
DEEPSORT_DIR = Path("outputs/deepsort_tracks")
IMAGE_ROOT = Path("data/kitti/tracking/training/image_02")
OUTPUT_DIR = Path("outputs/visualization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 2
BOX_THICKNESS = 2
GT_COLOR = (255, 255, 255)   # white for ground truth
TITLE_BAR_HEIGHT = 30        # pixels for the label bar in compare mode


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_color(track_id: int):
    """Deterministic per-track color."""
    np.random.seed(int(track_id) % (2**31 - 1))
    return tuple(int(c) for c in np.random.randint(50, 230, 3))


def draw_tracks(img, frame_tracks: pd.DataFrame):
    """Draw bounding boxes and labels for all tracks in a frame."""
    for _, row in frame_tracks.iterrows():
        x1, y1, x2, y2 = int(row.x1), int(row.y1), int(row.x2), int(row.y2)
        tid = int(row.track_id)
        color = get_color(tid)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        label = f"{row.class_name} #{tid}"
        # Dark background behind text for readability
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 3), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)
    return img


def draw_gt(img, gt_frame: pd.DataFrame):
    """Draw ground truth boxes in white (dashed style via dotted rectangle)."""
    for _, row in gt_frame.iterrows():
        x1, y1, x2, y2 = int(row.x1), int(row.y1), int(row.x2), int(row.y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, 1)
        cv2.putText(img, f"GT#{int(row.track_id)}", (x1, y2 + 12),
                    FONT, 0.4, GT_COLOR, 1)
    return img


def add_title_bar(img, title: str, color=(60, 60, 60)):
    """Add a colored title bar at the top of an image."""
    bar = np.full((TITLE_BAR_HEIGHT, img.shape[1], 3), color, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(title, FONT, 0.6, 2)
    tx = (img.shape[1] - tw) // 2
    ty = (TITLE_BAR_HEIGHT + th) // 2
    cv2.putText(bar, title, (tx, ty), FONT, 0.6, (255, 255, 255), 2)
    return np.vstack([bar, img])


def render_sequence(seq_name: str, mode: str, show_gt: bool):
    image_paths = sorted((IMAGE_ROOT / seq_name).glob("*.png"))
    if not image_paths:
        print(f"{seq_name}: no images found, skipping")
        return

    # Load track data
    sort_df = deepsort_df = None
    if mode in ("sort", "compare"):
        sort_file = SORT_DIR / f"{seq_name}.csv"
        if not sort_file.exists():
            print(f"{seq_name}: SORT tracks not found at {sort_file}")
            if mode == "sort":
                return
        else:
            sort_df = pd.read_csv(sort_file)

    if mode in ("deepsort", "compare"):
        ds_file = DEEPSORT_DIR / f"{seq_name}.csv"
        if not ds_file.exists():
            print(f"{seq_name}: Deep SORT tracks not found at {ds_file}")
            if mode == "deepsort":
                return
        else:
            deepsort_df = pd.read_csv(ds_file)

    # Load GT if needed
    gt_df = None
    if show_gt:
        try:
            gt_df = load_gt_for_sequence(seq_name)
        except FileNotFoundError:
            print(f"{seq_name}: GT not found, skipping GT overlay")

    # Determine output frame size
    first = cv2.imread(str(image_paths[0]))
    h, w = first.shape[:2]

    if mode == "compare":
        # Side-by-side: each panel gets a title bar, then they're placed side by side
        panel_h = h + TITLE_BAR_HEIGHT
        out_w = w * 2
        out_h = panel_h
    else:
        out_w = w
        out_h = h

    out_path = str(OUTPUT_DIR / f"{seq_name}_{mode}.mp4")
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (out_w, out_h)
    )

    print(f"Rendering {seq_name} [{mode}] → {out_path}")

    for img_path in image_paths:
        frame_id = int(img_path.stem)
        img = cv2.imread(str(img_path))

        gt_frame = gt_df[gt_df["frame"] == frame_id] if gt_df is not None else pd.DataFrame()

        if mode == "sort":
            ft = sort_df[sort_df["frame"] == frame_id] if sort_df is not None else pd.DataFrame()
            draw_tracks(img, ft)
            if not gt_frame.empty:
                draw_gt(img, gt_frame)
            writer.write(img)

        elif mode == "deepsort":
            ft = deepsort_df[deepsort_df["frame"] == frame_id] if deepsort_df is not None else pd.DataFrame()
            draw_tracks(img, ft)
            if not gt_frame.empty:
                draw_gt(img, gt_frame)
            writer.write(img)

        elif mode == "compare":
            left = img.copy()
            right = img.copy()

            if sort_df is not None:
                ft_sort = sort_df[sort_df["frame"] == frame_id]
                draw_tracks(left, ft_sort)
            if deepsort_df is not None:
                ft_ds = deepsort_df[deepsort_df["frame"] == frame_id]
                draw_tracks(right, ft_ds)
            if not gt_frame.empty:
                draw_gt(left, gt_frame)
                draw_gt(right, gt_frame)

            # Add title bars
            left = add_title_bar(left, "SORT (baseline)", color=(40, 40, 100))
            right = add_title_bar(right, "Deep SORT", color=(40, 100, 40))

            combined = np.hstack([left, right])
            writer.write(combined)

    writer.release()
    print(f"Saved to {out_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize MOT tracking results")
    parser.add_argument(
        "--mode",
        choices=["sort", "deepsort", "compare"],
        default="compare",
        help="Visualization mode (default: compare)",
    )
    parser.add_argument(
        "--show-gt",
        action="store_true",
        help="Overlay ground truth boxes in white",
    )
    parser.add_argument(
        "--seq",
        default=None,
        help="Run only this sequence (e.g. 0000). Defaults to all sequences.",
    )
    args = parser.parse_args()

    if args.seq:
        sequences = [args.seq]
    else:
        # Infer sequences from whichever track directory is most relevant
        if args.mode == "sort":
            sequences = [f.stem for f in sorted(SORT_DIR.glob("*.csv"))]
        elif args.mode == "deepsort":
            sequences = [f.stem for f in sorted(DEEPSORT_DIR.glob("*.csv"))]
        else:
            # compare: only sequences present in both
            sort_seqs = {f.stem for f in SORT_DIR.glob("*.csv")}
            ds_seqs = {f.stem for f in DEEPSORT_DIR.glob("*.csv")}
            sequences = sorted(sort_seqs & ds_seqs)

    if not sequences:
        print("No sequences found. Make sure track CSV files exist.")
        return

    for seq_name in sequences:
        render_sequence(seq_name, args.mode, args.show_gt)


if __name__ == "__main__":
    main()