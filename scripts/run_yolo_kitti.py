from pathlib import Path
import pandas as pd
import torch

# Change this to wherever the KITTI tracking images are stored locally.
# Example structure:
# data/kitti/tracking/training/image_02/0000/000000.png
KITTI_IMAGE_ROOT = Path("data/kitti/tracking/training/image_02")

# Detections will be saved here.
OUTPUT_DIR = Path("outputs/detections")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep the first pass simple.
# These are the two classes most relevant for your proposal.
KEEP_CLASSES = {"car", "person"}

# Load a pretrained YOLOv5 model.
# First run may download weights.
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# Simple thresholds for a first baseline.
model.conf = 0.25
model.iou = 0.45
model.eval()

sequence_dirs = sorted([p for p in KITTI_IMAGE_ROOT.iterdir() if p.is_dir()])

# For the first test, only run one sequence.
# Later you can remove [:1] and run all sequences.
for seq_dir in sequence_dirs[:1]:
    rows = []
    image_paths = sorted(seq_dir.glob("*.png"))

    print(f"Running YOLOv5 on sequence {seq_dir.name} with {len(image_paths)} frames")

    for img_path in image_paths:
        frame_id = int(img_path.stem)

        results = model(str(img_path))
        det_df = results.pandas().xyxy[0].copy()

        # Keep only car and person for now.
        det_df = det_df[det_df["name"].isin(KEEP_CLASSES)]

        for _, r in det_df.iterrows():
            rows.append(
                {
                    "sequence": seq_dir.name,
                    "frame": frame_id,
                    "x1": float(r["xmin"]),
                    "y1": float(r["ymin"]),
                    "x2": float(r["xmax"]),
                    "y2": float(r["ymax"]),
                    "confidence": float(r["confidence"]),
                    "class_id": int(r["class"]),
                    "class_name": str(r["name"]),
                }
            )

    out_file = OUTPUT_DIR / f"{seq_dir.name}.csv"
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(f"Saved detections to {out_file}")