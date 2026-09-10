#!/usr/bin/env python3
"""Run the validation-only RRWIT A0-A3 architecture ablation.

A0 is the unmodified COCO-pretrained YOLO11n baseline. A1 adds a P2-to-P3
assistance path, A2 adds CBAM at P3, and A3 combines both modifications. All
variants retain P3/P4/P5 detection outputs. The held-out test split is never
requested by this script; architecture screening is based on validation data.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import ultralytics.nn.tasks as yolo_tasks
import yaml
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

LOGGER = logging.getLogger('rrwit.ablation')
EXPERIMENTS = ('A0', 'A1', 'A2', 'A3')


class RRWITCBAM(nn.Module):
    """Convolutional Block Attention Module used in A2 and A3."""
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        maximum = torch.amax(x, dim=(2, 3), keepdim=True)
        x = x * self.sigmoid(self.channel_mlp(avg) + self.channel_mlp(maximum))
        spatial = torch.cat((torch.mean(x, dim=1, keepdim=True), torch.amax(x, dim=1, keepdim=True)), dim=1)
        return x * self.sigmoid(self.spatial(spatial))


def register_custom_module() -> None:
    yolo_tasks.RRWITCBAM = RRWITCBAM
    RRWITCBAM.__module__ = yolo_tasks.__name__


def architecture_yaml(experiment: str) -> dict[str, Any]:
    backbone = [
        [-1, 1, 'Conv', [64, 3, 2]], [-1, 1, 'Conv', [128, 3, 2]],
        [-1, 2, 'C3k2', [256, False, 0.25]], [-1, 1, 'Conv', [256, 3, 2]],
        [-1, 2, 'C3k2', [512, False, 0.25]], [-1, 1, 'Conv', [512, 3, 2]],
        [-1, 2, 'C3k2', [512, True]], [-1, 1, 'Conv', [1024, 3, 2]],
        [-1, 2, 'C3k2', [1024, True]], [-1, 1, 'SPPF', [1024, 5]],
        [-1, 2, 'C2PSA', [1024]],
    ]
    head = [
        [-1, 1, 'nn.Upsample', [None, 2, 'nearest']], [[-1, 6], 1, 'Concat', [1]],
        [-1, 2, 'C3k2', [512, False]], [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
        [[-1, 4], 1, 'Concat', [1]], [-1, 2, 'C3k2', [256, False]],
    ]
    if experiment in ('A1', 'A3'):
        head.extend([[2, 1, 'Conv', [256, 3, 2]], [[16, 17], 1, 'Concat', [1]], [-1, 1, 'C3k2', [256, False]]])
        p3 = 19
    else:
        p3 = 16
    if experiment in ('A2', 'A3'):
        head.append([p3, 1, 'RRWITCBAM', [64, 16, 7]])
        p3 += 1
    head.extend([
        [p3, 1, 'Conv', [256, 3, 2]], [[-1, 13], 1, 'Concat', [1]], [-1, 2, 'C3k2', [512, False]],
        [-1, 1, 'Conv', [512, 3, 2]], [[-1, 10], 1, 'Concat', [1]], [-1, 2, 'C3k2', [1024, True]],
    ])
    last = len(backbone) + len(head)
    head.append([[p3, last - 4, last - 1], 1, 'Detect', ['nc']])
    return {'nc': 14, 'scales': {'n': [0.50, 0.25, 1024]}, 'scale': 'n', 'backbone': backbone, 'head': head}


def source_layer(experiment: str, target_layer: int) -> int | None:
    if target_layer <= 16:
        return target_layer
    added = {'A1': 3, 'A2': 1, 'A3': 4}[experiment]
    first_base_after_p3 = 17 + added
    return None if target_layer < first_base_after_p3 else target_layer - added


def transfer_pretrained(model: YOLO, pretrained: str, experiment: str) -> int:
    """Transfer shape-compatible YOLO11n weights while respecting inserted modules."""
    source = YOLO(pretrained).model.state_dict()
    target = model.model.state_dict()
    transferred: dict[str, torch.Tensor] = {}
    pattern = re.compile(r'^model\.(\d+)\.(.+)$')
    for target_key, target_tensor in target.items():
        match = pattern.match(target_key)
        if match is None:
            continue
        layer = int(match.group(1))
        mapped = source_layer(experiment, layer) if experiment != 'A0' else layer
        if mapped is None:
            continue
        candidate = f'model.{mapped}.{match.group(2)}'
        if candidate in source and source[candidate].shape == target_tensor.shape:
            transferred[target_key] = source[candidate]
    model.model.load_state_dict(transferred, strict=False)
    return len(transferred)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='RRWIT validation-only A0-A3 architecture screen.')
    parser.add_argument('--experiment', choices=EXPERIMENTS, required=True)
    parser.add_argument('--data-yaml', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pretrained-weights', default='yolo11n.pt')
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    register_custom_module()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else 'cpu')

    if args.experiment == 'A0':
        model = YOLO(args.pretrained_weights)
        transferred = None
    else:
        model_yaml = architecture_yaml(args.experiment)
        yaml_path = args.output_dir / f'{args.experiment}_resolved_model.yaml'
        yaml_path.write_text(yaml.safe_dump(model_yaml, sort_keys=False))
        model = YOLO(str(yaml_path))
        transferred = transfer_pretrained(model, args.pretrained_weights, args.experiment)

    run = model.train(
        data=str(args.data_yaml), epochs=1000, imgsz=640, batch=16, workers=6,
        patience=15, device=device, cache=True, seed=args.seed, deterministic=True,
        amp=True, optimizer='auto', lr0=0.01, lrf=0.01, momentum=0.937,
        weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8,
        warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5,
        fliplr=0.5, mosaic=1.0, close_mosaic=10, mixup=0.0, copy_paste=0.0,
        project=str(args.output_dir), name=f'{args.experiment.lower()}_seed{args.seed}',
    )
    best = Path(run.save_dir) / 'weights' / 'best.pt'
    final_model = YOLO(str(best))
    metrics = final_model.val(data=str(args.data_yaml), split='val', imgsz=640, device=device)
    summary = {
        'experiment': args.experiment, 'seed': args.seed,
        'validation_mAP50': float(metrics.box.map50),
        'validation_mAP50_95': float(metrics.box.map),
        'parameter_count': int(sum(p.numel() for p in final_model.model.parameters())),
        'gflops_640': float(get_flops(final_model.model, imgsz=640)),
        'transferred_state_entries': transferred,
        'test_split_accessed': False,
    }
    (args.output_dir / f'{args.experiment}_validation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    LOGGER.info('%s', summary)


if __name__ == '__main__':
    main()
