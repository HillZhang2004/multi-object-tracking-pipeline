"""
CS153 Final Project: Multi-Object Tracking Pipeline
File: run_yolo_kitti.py
Authors: Spencer Merodio, Hill Zhang
Date: 2026-04-07

AI Use:
We used an AI assistant in a limited way on this script:
  - Reformatting comments and the file header so the script is easier to read.
  - Checking that the code structure matched the project plan from our proposal.
  - Sanity checking a few implementation details for loading a pretrained YOLOv5
    model, iterating through KITTI image sequences, and saving detections to CSV.

The project idea, pipeline design, and code organization came from our proposal
and our own implementation work. This script reflects the first detection stage
of the project: running a pretrained detector on KITTI frames and saving the
resulting detections for later tracking.

Summary:
This script runs a pretrained YOLOv5 detector on image sequences from the KITTI
tracking training set. For each frame, it saves:
  - bounding box coordinates
  - confidence score
  - class id
  - class name

For the first pass, the script keeps only the classes most relevant to our
project: car and person. The detections are written to CSV files in
outputs/detections/.

Usage:
  - Make sure KITTI tracking images are stored under:
        data/kitti/tracking/training/image_02/
  - Then run:
        python scripts/run_yolo_kitti.py
"""

from pathlib import Path
import pandas as pd
from ultralytics import YOLO

# Change this to wherever the KITTI tracking images are stored locally.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
KITTI_IMAGE_ROOT = PROJECT_ROOT / "data" / "kitti" / "tracking" / "training" / "image_02"

# Detections will be saved here.
OUTPUT_DIR = Path("outputs/detections")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep the first pass simple.
# These are the two classes most relevant for our proposal.
KEEP_CLASSES = {"car", "person"}

# Simple thresholds for a first baseline.
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Load a pretrained YOLOv5 model through the ultralytics package.
# First run will download the weights file (yolov5su.pt) into the local cache.
model = YOLO("yolov5su.pt")

sequence_dirs = sorted([p for p in KITTI_IMAGE_ROOT.iterdir() if p.is_dir()])

# Run the model on each sequence and save detections to CSV
for seq_dir in sequence_dirs:
    rows = []
    image_paths = sorted(seq_dir.glob("*.png"))

    print(f"Running YOLOv5 on sequence {seq_dir.name} with {len(image_paths)} frames")

    for img_path in image_paths:
        frame_id = int(img_path.stem)

        # ultralytics returns a list of Results, one per input image.
        results = model(str(img_path), conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        result = results[0]
        names = result.names  # class_id -> class_name

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, cls_ids):
            class_name = names[int(cid)]
            # Keep only car and person for now.
            if class_name not in KEEP_CLASSES:
                continue
            rows.append(
                {
                    "sequence": seq_dir.name,
                    "frame": frame_id,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "confidence": float(conf),
                    "class_id": int(cid),
                    "class_name": str(class_name),
                }
            )

    out_file = OUTPUT_DIR / f"{seq_dir.name}.csv"
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(f"Saved detections to {out_file}")