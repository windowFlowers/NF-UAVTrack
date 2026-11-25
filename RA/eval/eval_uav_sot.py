import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from eval.label_normalization import (
    load_prediction_file as load_canonical_prediction_file,
    load_xywh_labels,
)

ATTRIBUTE_SUBSETS = [
    "all",
    "night",
    "fog",
    "low_contrast",
    "tiny_3_8px",
    "tiny_8_16px",
    "tiny_16_32px",
    "occlusion",
    "fast_motion",
    "complex_background",
]

METRIC_COLUMNS = [
    "frames",
    "success_auc",
    "precision_20",
    "normalized_precision",
    "present_recall",
    "recall_iou_0.5",
    "recall_iou_0.3",
    "absent_accuracy",
    "false_positive_absent_rate",
    "miss_present_rate",
    "drift_rate",
    "recovery_time",
    "fps",
]


def compute_tracking_metrics(predictions, groundtruth, runtime_seconds=None):
    predictions = np.asarray(predictions, dtype=np.float32)
    groundtruth = np.asarray(groundtruth, dtype=np.float32)
    if predictions.ndim != 2 or predictions.shape[1] not in (4, 5):
        raise ValueError("predictions must have shape [N, 4] or [N, 5]")
    if groundtruth.ndim != 2 or groundtruth.shape[1] != 4:
        raise ValueError("groundtruth must have shape [N, 4]")
    if len(predictions) != len(groundtruth):
        raise ValueError("predictions and groundtruth must have the same number of frames")

    present_mask = _valid_boxes(groundtruth)
    absent_mask = ~present_mask
    pred_present = _valid_boxes(predictions[:, :4])
    if predictions.shape[1] == 5:
        pred_present = pred_present & (predictions[:, 4] > 0.0)

    ious = np.zeros(len(groundtruth), dtype=np.float32)
    center_errors = np.full(len(groundtruth), np.inf, dtype=np.float32)
    norm_errors = np.full(len(groundtruth), np.inf, dtype=np.float32)
    for idx in np.where(present_mask & pred_present)[0]:
        ious[idx] = _xywh_iou(predictions[idx, :4], groundtruth[idx])
        center_errors[idx] = _center_error(predictions[idx, :4], groundtruth[idx])
        norm_errors[idx] = center_errors[idx] / max(math.sqrt(float(groundtruth[idx, 2] * groundtruth[idx, 3])), 1.0)

    present_count = int(present_mask.sum())
    absent_count = int(absent_mask.sum())
    success_auc = _success_auc(ious[present_mask]) if present_count else 0.0
    precision_20 = float((center_errors[present_mask] <= 20.0).mean()) if present_count else 0.0
    normalized_precision = float((norm_errors[present_mask] <= 0.5).mean()) if present_count else 0.0
    present_recall = float(pred_present[present_mask].mean()) if present_count else 0.0
    recall_iou_05 = float((ious[present_mask] >= 0.5).mean()) if present_count else 0.0
    recall_iou_03 = float((ious[present_mask] >= 0.3).mean()) if present_count else 0.0
    absent_accuracy = float((~pred_present[absent_mask]).mean()) if absent_count else 0.0
    false_positive_absent_rate = float(pred_present[absent_mask].mean()) if absent_count else 0.0
    miss_present_rate = float((~pred_present[present_mask]).mean()) if present_count else 0.0
    drift_rate = _drift_rate(ious, present_mask, pred_present)
    recovery_time = _recovery_time(ious, present_mask, pred_present)
    fps = float(len(predictions) / runtime_seconds) if runtime_seconds and runtime_seconds > 0 else 0.0

    return {
        "frames": int(len(predictions)),
        "success_auc": success_auc,
        "precision_20": precision_20,
        "normalized_precision": normalized_precision,
        "present_recall": present_recall,
        "recall_iou_0.5": recall_iou_05,
        "recall_iou_0.3": recall_iou_03,
        "absent_accuracy": absent_accuracy,
        "false_positive_absent_rate": false_positive_absent_rate,
        "miss_present_rate": miss_present_rate,
        "drift_rate": drift_rate,
        "recovery_time": recovery_time,
        "fps": fps,
    }


def load_groundtruth_file(path):
    return load_xywh_labels(path, format="xywh_pixel", strict=True)


def load_prediction_file(path):
    return load_canonical_prediction_file(path, strict=True)


def load_attributes(path):
    attr_path = Path(path)
    if not attr_path.exists():
        return {}
    return json.loads(attr_path.read_text(encoding="utf-8"))


def evaluate_prediction_dir(dataset_root, prediction_dir, runtime_seconds=None, attribute_eval=True):
    dataset_root = Path(dataset_root)
    prediction_dir = Path(prediction_dir)
    per_sequence = {}
    sequence_records = []
    all_pred = []
    all_gt = []
    for seq_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        pred_path = prediction_dir / f"{seq_dir.name}.txt"
        if not pred_path.exists():
            continue
        gt = load_groundtruth_file(seq_dir / "groundtruth.txt")
        pred = load_prediction_file(pred_path)
        metrics = compute_tracking_metrics(pred, gt, runtime_seconds=None)
        per_sequence[seq_dir.name] = metrics
        attrs = load_attributes(seq_dir / "attributes.json")
        sequence_records.append({"name": seq_dir.name, "gt": gt, "pred": pred, "attributes": attrs})
        all_pred.append(pred)
        all_gt.append(gt)
    if not all_pred:
        raise FileNotFoundError(f"No prediction files found in {prediction_dir}")
    summary = compute_tracking_metrics(np.concatenate(all_pred), np.concatenate(all_gt), runtime_seconds=runtime_seconds)
    result = {"summary": summary, "sequences": per_sequence}
    if attribute_eval:
        result["attributes"] = evaluate_attribute_subsets(sequence_records)
    return result


def evaluate_attribute_subsets(sequence_records):
    attributes = {}
    for subset in ATTRIBUTE_SUBSETS:
        selected = [record for record in sequence_records if _matches_subset(record, subset)]
        if selected:
            pred = np.concatenate([record["pred"] for record in selected])
            gt = np.concatenate([record["gt"] for record in selected])
            metrics = compute_tracking_metrics(pred, gt)
        else:
            metrics = _empty_metrics()
        metrics["sequence_count"] = len(selected)
        attributes[subset] = metrics
    return attributes


def write_metric_tables(result, output_dir, method_name="method"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_rows = [{"method": method_name, **{key: result["summary"].get(key, 0) for key in METRIC_COLUMNS}}]
    _write_csv(output_dir / "main_results.csv", ["method", *METRIC_COLUMNS], main_rows)
    _write_markdown(output_dir / "main_results.md", ["method", *METRIC_COLUMNS], main_rows)

    attribute_rows = []
    for subset, metrics in result.get("attributes", {}).items():
        attribute_rows.append(
            {
                "method": method_name,
                "subset": subset,
                "sequence_count": metrics.get("sequence_count", 0),
                **{key: metrics.get(key, 0) for key in METRIC_COLUMNS},
            }
        )
    if attribute_rows:
        columns = ["method", "subset", "sequence_count", *METRIC_COLUMNS]
        _write_csv(output_dir / "attribute_results.csv", columns, attribute_rows)
        _write_markdown(output_dir / "attribute_results.md", columns, attribute_rows)


def _matches_subset(record, subset):
    if subset == "all":
        return True
    attrs = record.get("attributes") or {}
    if subset in {"night", "fog", "low_contrast", "complex_background"}:
        return bool(attrs.get("scene", {}).get(subset, False))
    if subset in {"occlusion", "fast_motion"}:
        return bool(attrs.get("motion", {}).get(subset, False))
    if subset.startswith("tiny_"):
        median_size = attrs.get("target", {}).get("median_size_px")
        if not isinstance(median_size, (int, float)):
            median_size = _median_gt_size(record["gt"])
        if subset == "tiny_3_8px":
            return 3 <= median_size < 8
        if subset == "tiny_8_16px":
            return 8 <= median_size < 16
        if subset == "tiny_16_32px":
            return 16 <= median_size <= 32
    return False


def _median_gt_size(groundtruth):
    valid = groundtruth[_valid_boxes(groundtruth)]
    if len(valid) == 0:
        return 0.0
    sizes = np.sqrt(valid[:, 2] * valid[:, 3])
    return float(np.median(sizes))


def _empty_metrics():
    return {
        "frames": 0,
        "success_auc": 0.0,
        "precision_20": 0.0,
        "normalized_precision": 0.0,
        "present_recall": 0.0,
        "recall_iou_0.5": 0.0,
        "recall_iou_0.3": 0.0,
        "absent_accuracy": 0.0,
        "false_positive_absent_rate": 0.0,
        "miss_present_rate": 0.0,
        "drift_rate": 0.0,
        "recovery_time": 0.0,
        "fps": 0.0,
    }


def _write_csv(path, columns, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_cell(row.get(column, "")) for column in columns})


def _write_markdown(path, columns, rows):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(column, "")) for column in columns) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_cell(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _valid_boxes(boxes):
    return np.isfinite(boxes).all(axis=1) & (boxes[:, 2] > 0) & (boxes[:, 3] > 0)


def _xywh_iou(box_a, box_b):
    ax1, ay1, aw, ah = [float(value) for value in box_a]
    bx1, by1, bw, bh = [float(value) for value in box_b]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _center_error(box_a, box_b):
    acx = float(box_a[0] + 0.5 * box_a[2])
    acy = float(box_a[1] + 0.5 * box_a[3])
    bcx = float(box_b[0] + 0.5 * box_b[2])
    bcy = float(box_b[1] + 0.5 * box_b[3])
    return math.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2)


def _success_auc(ious):
    if len(ious) == 0:
        return 0.0
    thresholds = np.linspace(0.0, 1.0, 21)
    return float(np.mean([(ious >= threshold).mean() for threshold in thresholds]))


def _drift_rate(ious, present_mask, pred_present, drift_iou_threshold=0.1):
    present_count = int(present_mask.sum())
    if present_count == 0:
        return 0.0
    drift = present_mask & pred_present & (ious < drift_iou_threshold)
    return float(drift.sum() / present_count)


def _recovery_time(ious, present_mask, pred_present, recovery_iou_threshold=0.5):
    recovery_times = []
    in_absent_gap = False
    frame_index = 0
    while frame_index < len(present_mask):
        if not present_mask[frame_index]:
            in_absent_gap = True
            frame_index += 1
            continue
        if not in_absent_gap:
            frame_index += 1
            continue

        start = frame_index
        while frame_index < len(present_mask) and present_mask[frame_index]:
            recovered = pred_present[frame_index] and ious[frame_index] >= recovery_iou_threshold
            if recovered:
                recovery_times.append(float(frame_index - start))
                break
            frame_index += 1
        else:
            recovery_times.append(float(frame_index - start))
            in_absent_gap = False
            continue

        while frame_index < len(present_mask) and present_mask[frame_index]:
            frame_index += 1
        in_absent_gap = False

    return float(np.mean(recovery_times)) if recovery_times else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate UAVTrack predictions under GT-init SOT protocol.")
    parser.add_argument("--dataset", required=True, help="Path to UAVTrack split, e.g. data/UAVTrack/test")
    parser.add_argument("--predictions", required=True, help="Directory containing seq_name.txt predictions")
    parser.add_argument("--runtime-seconds", type=float, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--tables-dir", default=None)
    parser.add_argument("--method-name", default="method")
    parser.add_argument("--no-attribute-eval", action="store_true")
    args = parser.parse_args()

    result = evaluate_prediction_dir(
        args.dataset,
        args.predictions,
        runtime_seconds=args.runtime_seconds,
        attribute_eval=not args.no_attribute_eval,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if args.tables_dir:
        write_metric_tables(result, args.tables_dir, method_name=args.method_name)
    print(text)


if __name__ == "__main__":
    main()
