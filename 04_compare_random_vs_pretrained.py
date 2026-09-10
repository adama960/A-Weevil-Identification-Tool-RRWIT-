#!/usr/bin/env python3
"""Run the matched RRWIT initialization experiment across three random seeds.

The experiment compares YOLO11n initialized from architecture-only YAML (random
weights) with independently initialized COCO-pretrained YOLO11n weights. Both
conditions use the same dataset partitions and training contract. Validation is
used for checkpoint selection; the held-out test split is evaluated only after
training has completed.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

LOGGER = logging.getLogger('rrwit.initialization')
SEEDS = (42, 123, 2025)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Matched random-vs-pretrained YOLO11n experiment.')
    parser.add_argument('--data-yaml', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pretrained-weights', default='yolo11n.pt')
    parser.add_argument('--model-yaml', default='yolo11n.yaml')
    parser.add_argument('--device', default=None)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--evaluate-test', action='store_true')
    return parser.parse_args()


def device_value(requested: str | None) -> str | int:
    if requested is not None:
        return requested
    return 0 if torch.cuda.is_available() else 'cpu'


def train_one(condition: str, seed: int, args: argparse.Namespace, device: str | int) -> dict[str, object]:
    if condition == 'random':
        model = YOLO(args.model_yaml)
    elif condition == 'pretrained':
        model = YOLO(args.pretrained_weights)
    else:
        raise ValueError(f'Unknown condition: {condition}')

    run_name = f'rrwit_{condition}_seed{seed}'
    results = model.train(
        data=str(args.data_yaml), epochs=1000, imgsz=640, batch=16,
        workers=args.workers, patience=15, device=device, cache=True,
        seed=seed, deterministic=True, amp=True, optimizer='auto',
        lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
        box=7.5, cls=0.5, dfl=1.5, dropout=0.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0,
        translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0,
        copy_paste=0.0, close_mosaic=10, multi_scale=False, rect=False,
        project=str(args.output_dir), name=run_name,
    )

    best = Path(results.save_dir) / 'weights' / 'best.pt'
    if not best.is_file():
        raise FileNotFoundError(f'Best checkpoint not found: {best}')

    row: dict[str, object] = {
        'condition': condition,
        'seed': seed,
        'best_checkpoint': str(best),
    }
    if args.evaluate_test:
        metrics = YOLO(str(best)).val(data=str(args.data_yaml), split='test', imgsz=640, device=device)
        row.update({
            'test_precision': float(metrics.box.mp),
            'test_recall': float(metrics.box.mr),
            'test_mAP50': float(metrics.box.map50),
            'test_mAP50_95': float(metrics.box.map),
        })
    return row


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    if not args.data_yaml.is_file():
        raise FileNotFoundError(args.data_yaml)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = device_value(args.device)

    rows = [train_one(condition, seed, args, device) for condition in ('random', 'pretrained') for seed in SEEDS]
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / 'matched_initialization_runs.csv', index=False)

    if args.evaluate_test:
        summary = frame.groupby('condition')[['test_mAP50', 'test_mAP50_95']].agg(['mean', 'std'])
        summary.to_csv(args.output_dir / 'matched_initialization_summary.csv')
        LOGGER.info('\n%s', summary)

    manifest = {
        'seeds': list(SEEDS),
        'conditions': ['random', 'pretrained'],
        'test_evaluated': bool(args.evaluate_test),
        'note': 'Validation-selected checkpoints are finalized before optional test evaluation.',
    }
    (args.output_dir / 'matched_initialization_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
