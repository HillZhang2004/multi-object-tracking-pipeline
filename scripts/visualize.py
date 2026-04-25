import cv2
import pandas as pd
import numpy as np
from pathlib import Path

TRACK_DIR = Path("outputs/sort_tracks")
IMAGE_ROOT = Path("/Users/spencermerodio/Downloads/data/kitti/tracking/training/image_02")
OUTPUT_DIR = Path("outputs/visualization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Different color per track ID
def get_color(track_id):
    np.random.seed(int(track_id))
    return tuple(np.random.randint(0, 255, 3).tolist())

for track_file in sorted(TRACK_DIR.glob("*.csv")):
    seq_name = track_file.stem
    track_df = pd.read_csv(track_file)
    image_paths = sorted((IMAGE_ROOT / seq_name).glob("*.png"))

    # Get frame size from first image
    first = cv2.imread(str(image_paths[0]))
    h, w = first.shape[:2]

    out_path = str(OUTPUT_DIR / f"{seq_name}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))

    for img_path in image_paths:
        frame_id = int(img_path.stem)
        img = cv2.imread(str(img_path))

        frame_tracks = track_df[track_df["frame"] == frame_id]
        for _, row in frame_tracks.iterrows():
            x1, y1, x2, y2 = int(row.x1), int(row.y1), int(row.x2), int(row.y2)
            tid = int(row.track_id)
            color = get_color(tid)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{row.class_name} #{tid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        writer.write(img)

    writer.release()
    print(f"Saved video to {out_path}")