"""
CS153 Final Project: Multi-Object Tracking Pipeline
File: evaluate_tracking.py
Authors: Spencer Merodio, Hill Zhang
Date: 2026-04-26

AI Use:
We used an AI assistant in a limited way on this script:
  - Reformatting comments and the file header so the script is easier to read.
  - Checking that the code structure matched the project plan from our proposal.
  - Helping sanity check a class-project implementation of MOT-style metrics.
  - Checking that the evaluation output format matched the final presentation needs.

The project idea, pipeline design, and code organization came from our proposal
and our own implementation work. This script implements the evaluation stage:
comparing SORT and Deep SORT tracks against KITTI ground truth labels using
MOT-style metrics.

Summary:
This script loads ground truth and predicted tracks for each sequence and
computes evaluation metrics for both the SORT baseline and Deep SORT tracker:

  - MOTA  (Multiple Object Tracking Accuracy): penalizes FP, FN, and ID switches
  - IDF1  (Identity F1): measures identity consistency over a sequence
  - HOTA  (Higher Order Tracking Accuracy): balances detection and association

This is a project-level evaluation script, not the official KITTI TrackEval
benchmark server. Results are printed per sequence and averaged across all
available evaluated sequences. A summary CSV is saved to
outputs/evaluation_results.csv.

Usage:
  - Make sure the following directories exist and contain CSV files:
        outputs/sort_tracks/
        outputs/deepsort_tracks/
  - Make sure KITTI ground truth labels are at:
        data/kitti/tracking/training/label_02/
  - Then run:
        python scripts/evaluate_tracking.py
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from load_kitti_gt import load_gt_for_sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SORT_DIR = PROJECT_ROOT / "outputs" / "sort_tracks"
DEEPSORT_DIR = PROJECT_ROOT / "outputs" / "deepsort_tracks"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "evaluation_results.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# IoU threshold for a detection to count as a true positive in MOTA / IDF1.
IOU_THRESHOLD = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# IoU helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(box_a, box_b):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter = max(0, x_b - x_a) * max(0, y_b - y_a)
    if inter == 0:
        return 0.0

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    denom = area_a + area_b - inter

    if denom <= 0:
        return 0.0

    return inter / denom


def match_detections(gt_boxes, pred_boxes, iou_thresh=IOU_THRESHOLD):
    """
    Match ground truth boxes to predicted boxes using the Hungarian algorithm.

    Returns:
      matched_gt, matched_pred, unmatched_gt, unmatched_pred

    The matched lists contain indices into the input lists.
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return [], [], list(range(len(gt_boxes))), list(range(len(pred_boxes)))

    cost = np.zeros((len(gt_boxes), len(pred_boxes)))

    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            cost[i, j] = 1.0 - compute_iou(gt_box, pred_box)

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_gt = []
    matched_pred = []

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= (1.0 - iou_thresh):
            matched_gt.append(r)
            matched_pred.append(c)

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    unmatched_pred = [j for j in range(len(pred_boxes)) if j not in matched_pred]

    return matched_gt, matched_pred, unmatched_gt, unmatched_pred


# ─────────────────────────────────────────────────────────────────────────────
# Per-sequence metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_mota_idf1(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """
    Compute MOTA and IDF1-style counts for one sequence and class.

    MOTA = 1 - (FP + FN + IDSW) / number of GT detections
    IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)

    Returns a dict with raw counts used by summarize().
    """
    frames = sorted(gt_df["frame"].unique())

    tp = fp = fn = idsw = 0

    # For IDF1, track the best-matching gt/pred identity pairs.
    # match_counts[gt_id][pred_id] = number of frames they were matched.
    match_counts = defaultdict(lambda: defaultdict(int))
    prev_gt_to_pred = {}

    for frame in frames:
        gt_frame = gt_df[gt_df["frame"] == frame]
        pred_frame = pred_df[pred_df["frame"] == frame]

        gt_boxes = gt_frame[["x1", "y1", "x2", "y2"]].values.tolist()
        gt_ids = gt_frame["track_id"].tolist()

        pred_boxes = pred_frame[["x1", "y1", "x2", "y2"]].values.tolist()
        pred_ids = pred_frame["track_id"].tolist()

        matched_gt, matched_pred, unmatched_gt, unmatched_pred = match_detections(
            gt_boxes,
            pred_boxes,
        )

        tp += len(matched_gt)
        fn += len(unmatched_gt)
        fp += len(unmatched_pred)

        curr_gt_to_pred = {}

        for gt_idx, pred_idx in zip(matched_gt, matched_pred):
            gt_id = gt_ids[gt_idx]
            pred_id = pred_ids[pred_idx]

            curr_gt_to_pred[gt_id] = pred_id
            match_counts[gt_id][pred_id] += 1

            if gt_id in prev_gt_to_pred and prev_gt_to_pred[gt_id] != pred_id:
                idsw += 1

        prev_gt_to_pred = curr_gt_to_pred

    num_gt_dets = len(gt_df)

    # IDF1: assign each GT identity to one predicted identity by maximizing
    # matched co-occurrence counts.
    gt_ids_all = list(match_counts.keys())
    pred_ids_all = list({p for gt in match_counts.values() for p in gt})

    if gt_ids_all and pred_ids_all:
        cost_idf = np.zeros((len(gt_ids_all), len(pred_ids_all)))

        for i, gt_id in enumerate(gt_ids_all):
            for j, pred_id in enumerate(pred_ids_all):
                cost_idf[i, j] = -match_counts[gt_id][pred_id]

        row_ind, col_ind = linear_sum_assignment(cost_idf)
        idtp = int(-cost_idf[row_ind, col_ind].sum())
    else:
        idtp = 0

    idfp = len(pred_df) - idtp
    idfn = len(gt_df) - idtp

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "IDSW": idsw,
        "num_gt_dets": num_gt_dets,
        "IDTP": idtp,
        "IDFP": idfp,
        "IDFN": idfn,
    }


def compute_hota(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    alpha_range=np.arange(0.05, 0.96, 0.05),
) -> float:
    """
    Compute a simplified project-level HOTA-style score.

    This averages sqrt(DetA * AssA) across IoU thresholds from 0.05 to 0.95.
    It is useful for comparing our two trackers under the same setup, but it
    should not be treated as the official KITTI TrackEval implementation.
    """
    hota_scores = []
    frames = sorted(gt_df["frame"].unique())

    for alpha in alpha_range:
        tp_dets = 0
        fp_dets = 0
        fn_dets = 0

        # assoc[gt_id][pred_id] = matched count at this IoU threshold.
        assoc = defaultdict(lambda: defaultdict(int))

        for frame in frames:
            gt_frame = gt_df[gt_df["frame"] == frame]
            pred_frame = pred_df[pred_df["frame"] == frame]

            gt_boxes = gt_frame[["x1", "y1", "x2", "y2"]].values.tolist()
            gt_ids = gt_frame["track_id"].tolist()

            pred_boxes = pred_frame[["x1", "y1", "x2", "y2"]].values.tolist()
            pred_ids = pred_frame["track_id"].tolist()

            matched_gt, matched_pred, unmatched_gt, unmatched_pred = match_detections(
                gt_boxes,
                pred_boxes,
                iou_thresh=alpha,
            )

            tp_dets += len(matched_gt)
            fn_dets += len(unmatched_gt)
            fp_dets += len(unmatched_pred)

            for gt_idx, pred_idx in zip(matched_gt, matched_pred):
                assoc[gt_ids[gt_idx]][pred_ids[pred_idx]] += 1

        total_dets = tp_dets + fp_dets + fn_dets

        if total_dets == 0:
            continue

        det_a = tp_dets / total_dets

        # Association score for matched GT/pred identity pairs.
        ass_scores = []

        for gt_id, pred_counts in assoc.items():
            for pred_id, tpa in pred_counts.items():
                fpa = sum(
                    assoc[other_gt][pred_id]
                    for other_gt in assoc
                    if other_gt != gt_id and pred_id in assoc[other_gt]
                )
                fna = sum(
                    count
                    for other_pred, count in pred_counts.items()
                    if other_pred != pred_id
                )

                denom = tpa + fpa + fna
                ass_j = tpa / denom if denom > 0 else 0.0
                ass_scores.append((tpa, ass_j))

        if ass_scores:
            total_tpa = sum(score[0] for score in ass_scores)
            ass_a = (
                sum(tpa * ass_j for tpa, ass_j in ass_scores) / total_tpa
                if total_tpa > 0 else 0.0
            )
        else:
            ass_a = 0.0

        hota_scores.append(np.sqrt(det_a * ass_a))

    return float(np.mean(hota_scores)) if hota_scores else 0.0


def summarize(counts: dict) -> dict:
    """Turn raw counts into MOTA and IDF1 scores."""
    num_gt = counts["num_gt_dets"]
    mota = 1.0 - (counts["FP"] + counts["FN"] + counts["IDSW"]) / max(num_gt, 1)

    denom_idf1 = 2 * counts["IDTP"] + counts["IDFP"] + counts["IDFN"]
    idf1 = 2 * counts["IDTP"] / max(denom_idf1, 1)

    return {"MOTA": round(mota, 4), "IDF1": round(idf1, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

CLASSES_TO_EVAL = ["car", "person"]
RESULT_COLUMNS = [
    "sequence",
    "class",
    "tracker",
    "MOTA",
    "IDF1",
    "HOTA",
    "IDSW",
    "TP",
    "FP",
    "FN",
]

results = []

sort_files = {f.stem: f for f in SORT_DIR.glob("*.csv")}
deepsort_files = {f.stem: f for f in DEEPSORT_DIR.glob("*.csv")}
sequences = sorted(sort_files.keys() & deepsort_files.keys())

if not sequences:
    raise RuntimeError("No sequences found with both SORT and Deep SORT tracks.")

for seq_name in sequences:
    print(f"\n=== Sequence {seq_name} ===")

    sort_df = pd.read_csv(sort_files[seq_name])
    deepsort_df = pd.read_csv(deepsort_files[seq_name])

    try:
        gt_df = load_gt_for_sequence(seq_name)
    except FileNotFoundError as error:
        print(f"  Skipping: {error}")
        continue

    for class_name in CLASSES_TO_EVAL:
        gt_cls = gt_df[gt_df["class_name"] == class_name]
        sort_cls = sort_df[sort_df["class_name"] == class_name]
        deepsort_cls = deepsort_df[deepsort_df["class_name"] == class_name]

        if gt_cls.empty:
            continue

        sort_counts = compute_mota_idf1(gt_cls, sort_cls)
        sort_scores = summarize(sort_counts)
        sort_hota = compute_hota(gt_cls, sort_cls)

        deepsort_counts = compute_mota_idf1(gt_cls, deepsort_cls)
        deepsort_scores = summarize(deepsort_counts)
        deepsort_hota = compute_hota(gt_cls, deepsort_cls)

        print(
            f"  [{class_name}]  SORT  → MOTA={sort_scores['MOTA']:.4f}  "
            f"IDF1={sort_scores['IDF1']:.4f}  HOTA={sort_hota:.4f}  "
            f"IDSW={sort_counts['IDSW']}"
        )
        print(
            f"  [{class_name}]  DSORT → MOTA={deepsort_scores['MOTA']:.4f}  "
            f"IDF1={deepsort_scores['IDF1']:.4f}  HOTA={deepsort_hota:.4f}  "
            f"IDSW={deepsort_counts['IDSW']}"
        )

        results.append(
            {
                "sequence": seq_name,
                "class": class_name,
                "tracker": "SORT",
                "MOTA": sort_scores["MOTA"],
                "IDF1": sort_scores["IDF1"],
                "HOTA": round(sort_hota, 4),
                "IDSW": sort_counts["IDSW"],
                "TP": sort_counts["TP"],
                "FP": sort_counts["FP"],
                "FN": sort_counts["FN"],
            }
        )

        results.append(
            {
                "sequence": seq_name,
                "class": class_name,
                "tracker": "DeepSORT",
                "MOTA": deepsort_scores["MOTA"],
                "IDF1": deepsort_scores["IDF1"],
                "HOTA": round(deepsort_hota, 4),
                "IDSW": deepsort_counts["IDSW"],
                "TP": deepsort_counts["TP"],
                "FP": deepsort_counts["FP"],
                "FN": deepsort_counts["FN"],
            }
        )

results_df = pd.DataFrame(results, columns=RESULT_COLUMNS)

if results_df.empty:
    print("No results to aggregate. Check that GT label files exist.")
else:
    print("\n=== Aggregate (mean across sequences) ===")
    agg = results_df.groupby(["tracker", "class"])[
        ["MOTA", "IDF1", "HOTA", "IDSW"]
    ].mean()
    print(agg.round(4).to_string())

    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved full results to {OUTPUT_FILE}")