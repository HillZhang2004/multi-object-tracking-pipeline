"""
CS153 Final Project: Multi-Object Tracking Pipeline
File: run_sort_kitti.py
Authors: Spencer Merodio, Hill Zhang
Date: 2026-04-07

AI Use:
We used an AI assistant in a limited way on this script:
  - Reformatting comments and the file header so the script is easier to read.
  - Checking that the code structure matched the project plan from our proposal.
  - Sanity checking a few implementation details for reading detection CSV files,
    passing detections into SORT, and saving motion-based tracks to CSV.

The project idea, pipeline design, and code organization came from our proposal
and our own implementation work. This script reflects the first tracking stage
of the project: taking saved detections and generating motion-based tracks using
SORT as a baseline.

Summary:
This script reads detection CSV files produced by run_yolo_kitti.py and uses
them as input to a SORT tracker. For each sequence, it:
  - loads frame-level detections
  - separates detections by class
  - updates the SORT tracker frame by frame
  - saves track ids and bounding boxes to CSV files in outputs/sort_tracks/

For the first pass, the script tracks the two classes most relevant to our
project: car and person.

Usage:
  - Make sure detection CSV files already exist in:
        outputs/detections/
  - Make sure the SORT code is available under:
        third_party/sort/
  - Then run:
        python scripts/run_sort_kitti.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Put the SORT code in third_party/sort first.
# Sort isnt pip installed, so we have to add it to the path manually.
sys.path.append("third_party/sort")
from sort import Sort

# detections come from the yolo script, tracks go into sort_tracks
DET_DIR = Path("outputs/detections")
TRACK_DIR = Path("outputs/sort_tracks")
TRACK_DIR.mkdir(parents=True, exist_ok=True)

# Classes we want to track with SORT
# Same as with yolo script
CLASSES_TO_TRACK = ["car", "person"]

for det_file in sorted(DET_DIR.glob("*.csv")):
    seq_name = det_file.stem
    det_df = pd.read_csv(det_file)

    # Skip empty sequences
    if det_df.empty:
        print(f"{seq_name}: no detections found")
        continue

    max_frame = int(det_df["frame"].max())
    all_tracks = []

    print(f"Running SORT on sequence {seq_name}")
    
    # Run SORT separately for each class. This is a simple way to avoid ID switches
    for class_name in CLASSES_TO_TRACK:
        class_df = det_df[det_df["class_name"] == class_name].copy()

        # Each class gets a tracker
        # SORT tracker: Bewley et al., "Simple Online and Realtime Tracking," ICIP 2016.
        # Reference implementation: https://github.com/abewley/sort
        tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

        # Update the tracker frame by frame
        for frame in range(max_frame + 1):
            frame_df = class_df[class_df["frame"] == frame]

            if len(frame_df) > 0:
                dets = frame_df[["x1", "y1", "x2", "y2", "confidence"]].to_numpy(dtype=float)
            else:
                dets = np.empty((0, 5), dtype=float)

            # tracker.update() expects [x1,y1,x2,y2,conf] and returns [x1,y1,x2,y2,track_id].
            # Output format defined by abewley/sort.
            tracks = tracker.update(dets)

            for trk in tracks:
                x1, y1, x2, y2, track_id = trk
                all_tracks.append(
                    {
                        "sequence": seq_name,
                        "frame": frame,
                        "track_id": int(track_id),
                        "class_name": class_name,
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    }
                )

    # export all tracks for this sequence to a CSV file
    out_file = TRACK_DIR / f"{seq_name}.csv"
    pd.DataFrame(all_tracks).to_csv(out_file, index=False)
    print(f"Saved SORT tracks to {out_file}")