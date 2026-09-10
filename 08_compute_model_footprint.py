#!/usr/bin/env python3
"""Measure model parameter footprints and theoretical forward-operation counts.

This script is reporting-only: it does not train, tune, quantize, or evaluate
validation/test accuracy. Measurements should be interpreted jointly with each
framework's frozen input resolution; theoretical FLOPs are not runtime latency.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

MODEL_INPUTS = {
    'yolov5nu': 640, 'yolov8n': 640, 'yolo11n_rrwit': 640, 'yolo12n': 640, 'yolo26n': 640,
    'ssdlite320_mobilenet_v3_large': 320, 'efficientdet_d0': 512,
    'fasterrcnn_mobilenet_v3_large_320_fpn': 320,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Measure one canonical benchmark model footprint.')
    parser.add_argument('--model-name', choices=tuple(MODEL_INPUTS), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--loader-script', type=Path, required=True,
                        help='Module defining load_canonical_model(model_name, checkpoint).')
    parser.add_argument('--output-json', type=Path, required=True)
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location('rrwit_public_model_loader', path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    module = load_module(args.loader_script)
    model = module.load_canonical_model(args.model_name, args.checkpoint)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    input_px = MODEL_INPUTS[args.model_name]
    example = torch.zeros((1, 3, input_px, input_px), dtype=torch.float32)
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        _ = model(example)
    flops = int(counter.get_total_flops())
    result = {
        'model': args.model_name, 'input_px': input_px,
        'parameter_count': int(params), 'parameter_millions': params / 1e6,
        'fp32_parameter_memory_mib': param_bytes / (1024 ** 2),
        'forward_gflops': flops / 1e9,
        'interpretation': 'Theoretical batch-1 forward operations; not measured latency.',
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
