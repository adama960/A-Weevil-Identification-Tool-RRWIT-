#!/usr/bin/env python3
"""Export the four RRWIT deployment candidates to static FP32 ONNX artifacts.

The retained deployment candidates are YOLO26n, YOLO12n, RRWIT (YOLO11n), and
EfficientDet-D0. Export uses batch size 1 and opset 17. This stage performs no
accuracy evaluation, threshold tuning, quantization, or runtime benchmarking.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import onnx
from ultralytics import YOLO

CANDIDATES = {'yolo26n': 640, 'yolo12n': 640, 'yolo11n_rrwit': 640, 'efficientdet_d0': 512}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export one RRWIT deployment candidate to FP32 ONNX.')
    parser.add_argument('--model-name', choices=tuple(CANDIDATES), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--efficientdet-exporter', type=Path,
                        help='Module defining export_efficientdet_fp32(checkpoint, output_path, input_px, opset).')
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location('rrwit_effdet_exporter', path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    input_px = CANDIDATES[args.model_name]
    out = args.output_dir / f'{args.model_name}_FP32_opset17.onnx'
    if args.model_name != 'efficientdet_d0':
        exported = YOLO(str(args.checkpoint)).export(
            format='onnx', imgsz=input_px, batch=1, opset=17,
            dynamic=False, simplify=False, nms=False, half=False,
        )
        Path(exported).replace(out)
    else:
        if args.efficientdet_exporter is None:
            raise ValueError('--efficientdet-exporter is required for EfficientDet-D0.')
        module = load_module(args.efficientdet_exporter)
        module.export_efficientdet_fp32(args.checkpoint, out, input_px=input_px, opset=17)
    graph = onnx.load(str(out)); onnx.checker.check_model(graph)
    info = {'model': args.model_name, 'input_px': input_px, 'opset': 17, 'onnx_path': str(out), 'bytes': out.stat().st_size}
    (args.output_dir / f'{args.model_name}_FP32_export.json').write_text(json.dumps(info, indent=2) + '\n')


if __name__ == '__main__':
    main()
