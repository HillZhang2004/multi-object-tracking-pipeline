"""
CS153 Final Project: Multi-Object Tracking Pipeline
File: evaluate_tracking.py
Authors: Spencer Merodio, Hill Zhang
Date: 2026-04-07

AI Use:
We used an AI assistant in a limited way on this script:
  - Reformatting comments and the file header so the script is easier to read.
  - Checking that the code structure matched the project plan from our proposal.
  - Sanity checking the MOTA, IDF1, and HOTA formula implementations against
    the definitions in the papers cited in our proposal.

The project idea, pipeline design, and code organization came from our proposal
and our own implementation work. This script implements the evaluation stage:
comparing SORT and Deep SORT tracks against KITTI ground truth using the
metrics described in our proposal (MOTA, IDF1, HOTA).

Summary:
This script loads ground truth and predicted tracks for each sequence and
computes standard MOT metrics for both the SORT baseline and Deep SORT tracker:

  - MOTA  (Multiple Object Tracking Accuracy): penalizes FP, FN, ID switches
  - IDF1  (Identity F1): measures identity consistency over full sequences
  - HOTA  (Higher Order Tracking Accuracy): joint detection + association score

Results are printed per-sequence and averaged across all sequences. A summary
CSV is saved to outputs/evaluation_results.csv.

Usage:
  - Make sure the following directories exist and contain CSV files:
        outputs/sort_tracks/
        outputs/deepsort_tracks/
  - Make sure KITTI ground truth labels are at:
        data/kitti/tracking/training/label_02/
  - Then run:
        python scripts/evaluate_tracking.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from load_kitti_gt import load_gt_for_sequence

SORT_DIR = Path("outputs/sort_tracks")
DEEPSORT_DIR = Path("outputs/deepsort_tracks")
OUTPUT_FILE = Path("outputs/evaluation_results.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# IoU threshold for a detection to count as a true positive
IOU_THRESHOLD = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# IoU helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(boxA, boxB):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


def match_detections(gt_boxes, pred_boxes, iou_thresh=IOU_THRESHOLD):
    """
    Match ground truth boxes to predicted boxes using the Hungarian algorithm.
    Returns (matched_gt_ids, matched_pred_ids, unmatched_gt, unmatched_pred).
    Indices are into the input lists.
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return [], [], list(range(len(gt_boxes))), list(range(len(pred_boxes)))

    cost = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            cost[i, j] = 1.0 - compute_iou(g, p)

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_gt, matched_pred = [], []
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
    Compute MOTA and IDF1 for one sequence and class.

    MOTA = 1 - (FP + FN + IDSW) / num_gt_objects
    IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)

    Returns a dict with keys: TP, FP, FN, IDSW, num_gt, IDTP, IDFP, IDFN
    """
    frames = sorted(gt_df["frame"].unique())

    TP = FP = FN = IDSW = 0
    # For IDF1 we track the best-matching gt↔pred identity pairs.
    # count[gt_id][pred_id] = number of frames they were matched
    from collections import defaultdict
    match_counts = defaultdict(lambda: defaultdict(int))
    prev_gt_to_pred = {}  # gt_id -> pred_id from previous frame

    for frame in frames:
        gt_frame = gt_df[gt_df["frame"] == frame]
        pred_frame = pred_df[pred_df["frame"] == frame]

        gt_boxes = gt_frame[["x1", "y1", "x2", "y2"]].values.tolist()
        gt_ids = gt_frame["track_id"].tolist()
        pred_boxes = pred_frame[["x1", "y1", "x2", "y2"]].values.tolist()
        pred_ids = pred_frame["track_id"].tolist()

        matched_gt, matched_pred, unmatched_gt, unmatched_pred = match_detections(
            gt_boxes, pred_boxes
        )

        TP += len(matched_gt)
        FN += len(unmatched_gt)
        FP += len(unmatched_pred)

        # Count ID switches: a gt object that was matched to a *different*
        # pred ID compared to the previous frame.
        curr_gt_to_pred = {}
        for gi, pi in zip(matched_gt, matched_pred):
            gt_id = gt_ids[gi]
            pred_id = pred_ids[pi]
            curr_gt_to_pred[gt_id] = pred_id
            match_counts[gt_id][pred_id] += 1
            if gt_id in prev_gt_to_pred and prev_gt_to_pred[gt_id] != pred_id:
                IDSW += 1

        prev_gt_to_pred = curr_gt_to_pred

    num_gt = (gt_df["track_id"].nunique() * len(frames))  # approx total GT detections
    num_gt_dets = len(gt_df)

    # IDF1 — find optimal gt↔pred identity assignment by maximizing co-occurrence
    gt_ids_all = list(match_counts.keys())
    pred_ids_all = list({p for gt in match_counts.values() for p in gt})
    if gt_ids_all and pred_ids_all:
        cost_idf = np.zeros((len(gt_ids_all), len(pred_ids_all)))
        for i, g in enumerate(gt_ids_all):
            for j, p in enumerate(pred_ids_all):
                cost_idf[i, j] = -match_counts[g][p]  # maximize co-occurrence
        row_ind, col_ind = linear_sum_assignment(cost_idf)
        IDTP = int(-cost_idf[row_ind, col_ind].sum())
    else:
        IDTP = 0

    # IDFP: predicted frames not matched to any GT identity
    IDFP = len(pred_df) - IDTP
    # IDFN: GT frames not matched to any predicted identity
    IDFN = len(gt_df) - IDTP

    return {
        "TP": TP, "FP": FP, "FN": FN, "IDSW": IDSW,
        "num_gt_dets": num_gt_dets,
        "IDTP": IDTP, "IDFP": IDFP, "IDFN": IDFN,
    }


def compute_hota(gt_df: pd.DataFrame, pred_df: pd.DataFrame,
                 alpha_range=np.arange(0.05, 0.96, 0.05)) -> float:
    """
    Compute HOTA averaged over IoU thresholds from 0.05 to 0.95 in steps of 0.05.
    HOTA(alpha) = sqrt(DetA(alpha) * AssA(alpha))
    HOTA = mean over alpha of HOTA(alpha)
    """
    hota_scores = []
    frames = sorted(gt_df["frame"].unique())

    for alpha in alpha_range:
        # Per-frame matching at this alpha
        # For AssA we need to track which gt/pred IDs were matched together
        tp_dets = 0
        fp_dets = 0
        fn_dets = 0

        # association counts: how often gt_id matched pred_id at this alpha
        from collections import defaultdict
        assoc = defaultdict(lambda: defaultdict(int))

        for frame in frames:
            gt_frame = gt_df[gt_df["frame"] == frame]
            pred_frame = pred_df[pred_df["frame"] == frame]

            gt_boxes = gt_frame[["x1", "y1", "x2", "y2"]].values.tolist()
            gt_ids = gt_frame["track_id"].tolist()
            pred_boxes = pred_frame[["x1", "y1", "x2", "y2"]].values.tolist()
            pred_ids = pred_frame["track_id"].tolist()

            matched_gt, matched_pred, unmatched_gt, unmatched_pred = match_detections(
                gt_boxes, pred_boxes, iou_thresh=alpha
            )

            tp_dets += len(matched_gt)
            fn_dets += len(unmatched_gt)
            fp_dets += len(unmatched_pred)

            for gi, pi in zip(matched_gt, matched_pred):
                assoc[gt_ids[gi]][pred_ids[pi]] += 1

        total_dets = tp_dets + fp_dets + fn_dets
        if total_dets == 0:
            continue
        det_a = tp_dets / (tp_dets + fp_dets + fn_dets)

        # AssA: for each matched (gt, pred) pair, compute jaccard of their
        # co-occurrence counts vs. their individual totals.
        ass_scores = []
        for gt_id, pred_counts in assoc.items():
            for pred_id, tpa in pred_counts.items():
                # TPA = co-occurrence
                # FPA = frames pred_id was matched to anything other than gt_id
                # FNA = frames gt_id was matched to anything other than pred_id
                fpa = sum(
                    cnt for pid, cnt in pred_counts.items() if pid != pred_id
                ) + sum(
                    assoc[gid][pred_id]
                    for gid in assoc if gid != gt_id and pred_id in assoc[gid]
                )
                fna = sum(cnt for pid, cnt in pred_counts.items() if pid != pred_id)
                ass_j = tpa / (tpa + fpa + fna) if (tpa + fpa + fna) > 0 else 0.0
                ass_scores.append((tpa, ass_j))

        if ass_scores:
            total_tpa = sum(s[0] for s in ass_scores)
            ass_a = (
                sum(s[0] * s[1] for s in ass_scores) / total_tpa
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
    except FileNotFoundError as e:
        print(f"  Skipping: {e}")
        continue

    for cls in CLASSES_TO_EVAL:
        gt_cls = gt_df[gt_df["class_name"] == cls]
        sort_cls = sort_df[sort_df["class_name"] == cls]
        deepsort_cls = deepsort_df[deepsort_df["class_name"] == cls]

        if gt_cls.empty:
            continue

        # SORT metrics
        sort_counts = compute_mota_idf1(gt_cls, sort_cls)
        sort_scores = summarize(sort_counts)
        sort_hota = compute_hota(gt_cls, sort_cls)

        # Deep SORT metrics
        ds_counts = compute_mota_idf1(gt_cls, deepsort_cls)
        ds_scores = summarize(ds_counts)
        ds_hota = compute_hota(gt_cls, deepsort_cls)

        print(
            f"  [{cls}]  SORT  → MOTA={sort_scores['MOTA']:.4f}  "
            f"IDF1={sort_scores['IDF1']:.4f}  HOTA={sort_hota:.4f}  "
            f"IDSW={sort_counts['IDSW']}"
        )
        print(
            f"  [{cls}]  DSORT → MOTA={ds_scores['MOTA']:.4f}  "
            f"IDF1={ds_scores['IDF1']:.4f}  HOTA={ds_hota:.4f}  "
            f"IDSW={ds_counts['IDSW']}"
        )

        results.append({
            "sequence": seq_name, "class": cls, "tracker": "SORT",
            "MOTA": sort_scores["MOTA"], "IDF1": sort_scores["IDF1"],
            "HOTA": round(sort_hota, 4), "IDSW": sort_counts["IDSW"],
            "TP": sort_counts["TP"], "FP": sort_counts["FP"], "FN": sort_counts["FN"],
        })
        results.append({
            "sequence": seq_name, "class": cls, "tracker": "DeepSORT",
            "MOTA": ds_scores["MOTA"], "IDF1": ds_scores["IDF1"],
            "HOTA": round(ds_hota, 4), "IDSW": ds_counts["IDSW"],
            "TP": ds_counts["TP"], "FP": ds_counts["FP"], "FN": ds_counts["FN"],
        })

results_df = pd.DataFrame(results)

if results_df.empty:
    print("No results to aggregate — check that GT label files exist.")
else:
    print("\n=== Aggregate (mean across sequences) ===")
    agg = results_df.groupby(["tracker", "class"])[["MOTA", "IDF1", "HOTA", "IDSW"]].mean()
    print(agg.round(4).to_string())
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved full results to {OUTPUT_FILE}")

# Print aggregate summary
print("\n=== Aggregate (mean across sequences) ===")
agg = results_df.groupby(["tracker", "class"])[["MOTA", "IDF1", "HOTA", "IDSW"]].mean()
print(agg.round(4).to_string())

results_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved full results to {OUTPUT_FILE}")