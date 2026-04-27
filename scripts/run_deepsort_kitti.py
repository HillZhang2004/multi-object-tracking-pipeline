"""
CS153 Final Project: Multi-Object Tracking Pipeline
File: run_deepsort_kitti.py
Authors: Spencer Merodio, Hill Zhang
Date: 2026-04-07

AI Use:
We used an AI assistant in a limited way on this script:
  - Reformatting comments and the file header so the script is easier to read.
  - Checking that the code structure matched the project plan from our proposal.
  - Sanity checking a few implementation details for reading detection CSV files,
    passing detections into Deep SORT, and saving appearance-based tracks to CSV.

The project idea, pipeline design, and code organization came from our proposal
and our own implementation work. This script reflects the second tracking stage
of the project: taking saved detections and generating appearance-aware tracks
using Deep SORT.

Summary:
This script reads detection CSV files produced by run_yolo_kitti.py and uses
them as input to a Deep SORT tracker. For each sequence, it:
  - loads frame-level detections
  - loads the corresponding KITTI images (needed for the appearance encoder)
  - separates detections by class
  - updates the Deep SORT tracker frame by frame
  - saves track ids and bounding boxes to CSV files in outputs/deepsort_tracks/

For the first pass, the script tracks the two classes most relevant to our
project: car and person.

Usage:
  - Make sure detection CSV files already exist in:
        outputs/detections/
  - Make sure deep-sort-realtime is installed:
        pip install deep-sort-realtime
  - Then run:
        python scripts/run_deepsort_kitti.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import cv2

from deep_sort_realtime.deepsort_tracker import DeepSort

# detections come from the yolo script, tracks go into deepsort_tracks
DET_DIR = Path("outputs/detections")
TRACK_DIR = Path("outputs/deepsort_tracks")
TRACK_DIR.mkdir(parents=True, exist_ok=True)

# Path to KITTI images — needed by the appearance encoder
KITTI_IMAGE_ROOT = Path("data/kitti/tracking/training/image_02")

# Classes we want to track with Deep SORT
# Same as with the SORT script
CLASSES_TO_TRACK = ["car", "person"]

# Deep SORT hyperparameters
MAX_COSINE_DISTANCE = 0.4
MAX_AGE = 5
N_INIT = 3

for det_file in sorted(DET_DIR.glob("*.csv")):
    seq_name = det_file.stem
    det_df = pd.read_csv(det_file)

    # Skip empty sequences
    if det_df.empty:
        print(f"{seq_name}: no detections found")
        continue

    max_frame = int(det_df["frame"].max())
    all_tracks = []

    print(f"Running Deep SORT on sequence {seq_name}")

    # Run Deep SORT separately for each class. Same approach as SORT baseline.
    for class_name in CLASSES_TO_TRACK:
        class_df = det_df[det_df["class_name"] == class_name].copy()

        # Each class gets a tracker
        tracker = DeepSort(
            max_age=MAX_AGE,
            n_init=N_INIT,
            max_cosine_distance=MAX_COSINE_DISTANCE,
        )

        # Update the tracker frame by frame
        for frame in range(max_frame + 1):
            frame_df = class_df[class_df["frame"] == frame]

            # Load the image so the appearance encoder can crop detections
            img_path = KITTI_IMAGE_ROOT / seq_name / f"{frame:06d}.png"
            img_bgr = cv2.imread(str(img_path))

            if len(frame_df) > 0 and img_bgr is not None:
                # deep-sort-realtime expects a list of ([x,y,w,h], confidence, class) tuples
                x1 = frame_df["x1"].to_numpy(dtype=float)
                y1 = frame_df["y1"].to_numpy(dtype=float)
                w  = (frame_df["x2"] - frame_df["x1"]).to_numpy(dtype=float)
                h  = (frame_df["y2"] - frame_df["y1"]).to_numpy(dtype=float)
                confidences = frame_df["confidence"].to_numpy(dtype=float)

                raw_dets = [
                    ([x1[i], y1[i], w[i], h[i]], confidences[i], class_name)
                    for i in range(len(frame_df))
                ]
            else:
                raw_dets = []

            if img_bgr is None:
                print(f"  Warning: could not read image {img_path}, skipping frame {frame}")
                continue

            # returns a list of Track objects
            tracks = tracker.update_tracks(raw_dets, frame=img_bgr)

            for trk in tracks:
                if not trk.is_confirmed():
                    continue

                # convert back to [x1, y1, x2, y2] to match the SORT output format
                x1_out, y1_out, x2_out, y2_out = trk.to_ltrb()

                all_tracks.append(
                    {
                        "sequence": seq_name,
                        "frame": frame,
                        "track_id": int(trk.track_id),
                        "class_name": class_name,
                        "x1": float(x1_out),
                        "y1": float(y1_out),
                        "x2": float(x2_out),
                        "y2": float(y2_out),
                    }
                )

    # export all tracks for this sequence to a CSV file
    out_file = TRACK_DIR / f"{seq_name}.csv"
    pd.DataFrame(all_tracks).to_csv(out_file, index=False)
    print(f"Saved Deep SORT tracks to {out_file}")