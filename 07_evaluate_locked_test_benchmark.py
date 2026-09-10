#!/usr/bin/env python3
"""Evaluate a validation-locked RRWIT benchmark without test-driven reselection.

The script expects framework-native inference to have been standardized to COCO
prediction JSON. It selects an F1 operating threshold on validation predictions,
locks that threshold, then evaluates the corresponding frozen test predictions.
Checkpoint selection must already have been completed using validation mAP50-95.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validation-lock then test-evaluate one RRWIT benchmark task.')
    parser.add_argument('--val-annotations', type=Path, required=True)
    parser.add_argument('--val-predictions', type=Path, required=True)
    parser.add_argument('--test-annotations', type=Path, required=True)
    parser.add_argument('--test-predictions', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    return parser.parse_args()


def coco_metrics(annotation_path: Path, predictions: list[dict], threshold: float | None = None) -> dict[str, float]:
    gt = COCO(str(annotation_path))
    pred = predictions if threshold is None else [p for p in predictions if float(p.get('score', 0.0)) >= threshold]
    dt = gt.loadRes(pred) if pred else gt.loadRes([])
    ev = COCOeval(gt, dt, 'bbox')
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate(); ev.accumulate()
    return {
        'mAP50_95': float(ev.stats[0]), 'mAP50': float(ev.stats[1]),
        'mAP75': float(ev.stats[2]), 'AR100': float(ev.stats[8]),
    }


def f1_at_threshold(annotation_path: Path, predictions: list[dict], threshold: float, match_iou: float = 0.50) -> float:
    gt = COCO(str(annotation_path))
    by_image_gt: dict[int, list[dict]] = {}
    for ann in gt.dataset['annotations']:
        by_image_gt.setdefault(int(ann['image_id']), []).append(ann)
    by_image_pred: dict[int, list[dict]] = {}
    for pred in predictions:
        if float(pred.get('score', 0.0)) >= threshold:
            by_image_pred.setdefault(int(pred['image_id']), []).append(pred)

    tp = fp = fn = 0
    for image_id in gt.getImgIds():
        targets = by_image_gt.get(int(image_id), [])
        detections = sorted(by_image_pred.get(int(image_id), []), key=lambda x: float(x['score']), reverse=True)
        used: set[int] = set()
        for det in detections:
            dx, dy, dw, dh = map(float, det['bbox'])
            best_iou, best_j = 0.0, None
            for j, ann in enumerate(targets):
                if j in used or int(ann['category_id']) != int(det['category_id']):
                    continue
                gx, gy, gw, gh = map(float, ann['bbox'])
                ix1, iy1 = max(dx, gx), max(dy, gy)
                ix2, iy2 = min(dx + dw, gx + gw), min(dy + dh, gy + gh)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = dw * dh + gw * gh - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j is not None and best_iou >= match_iou:
                used.add(best_j); tp += 1
            else:
                fp += 1
        fn += len(targets) - len(used)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def select_validation_threshold(annotation_path: Path, predictions: list[dict]) -> tuple[float, float]:
    scores = np.array(sorted({float(p['score']) for p in predictions}), dtype=float)
    if scores.size == 0:
        return 1.0, 0.0
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, min(201, scores.size))))
    evaluated = [(float(t), f1_at_threshold(annotation_path, predictions, float(t))) for t in candidates]
    return max(evaluated, key=lambda item: (item[1], item[0]))


def main() -> None:
    args = parse_args()
    val_preds = json.loads(args.val_predictions.read_text())
    test_preds = json.loads(args.test_predictions.read_text())

    threshold, val_f1 = select_validation_threshold(args.val_annotations, val_preds)
    result = {
        'locked_confidence_threshold': threshold,
        'validation_weighted_global_f1_proxy': val_f1,
        'validation_coco_metrics': coco_metrics(args.val_annotations, val_preds),
        'test_coco_metrics': coco_metrics(args.test_annotations, test_preds),
        'test_f1_at_validation_locked_threshold': f1_at_threshold(args.test_annotations, test_preds, threshold),
        'test_used_for_threshold_selection': False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
