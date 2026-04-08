from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Put the SORT code in third_party/sort first.
sys.path.append("third_party/sort")
from sort import Sort

DET_DIR = Path("outputs/detections")
TRACK_DIR = Path("outputs/sort_tracks")
TRACK_DIR.mkdir(parents=True, exist_ok=True)

CLASSES_TO_TRACK = ["car", "person"]

for det_file in sorted(DET_DIR.glob("*.csv")):
    seq_name = det_file.stem
    det_df = pd.read_csv(det_file)

    if det_df.empty:
        print(f"{seq_name}: no detections found")
        continue

    max_frame = int(det_df["frame"].max())
    all_tracks = []

    print(f"Running SORT on sequence {seq_name}")

    for class_name in CLASSES_TO_TRACK:
        class_df = det_df[det_df["class_name"] == class_name].copy()

        tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

        for frame in range(max_frame + 1):
            frame_df = class_df[class_df["frame"] == frame]

            if len(frame_df) > 0:
                dets = frame_df[["x1", "y1", "x2", "y2", "confidence"]].to_numpy(dtype=float)
            else:
                dets = np.empty((0, 5), dtype=float)

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

    out_file = TRACK_DIR / f"{seq_name}.csv"
    pd.DataFrame(all_tracks).to_csv(out_file, index=False)
    print(f"Saved SORT tracks to {out_file}")