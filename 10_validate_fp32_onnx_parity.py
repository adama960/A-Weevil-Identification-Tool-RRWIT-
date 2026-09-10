#!/usr/bin/env python3
"""Validate FP32 ONNX parity against a canonical PyTorch model on validation data.

This script is validation-only and does not access the locked test split. A
public model-specific adapter supplies deterministic preprocessing, canonical
PyTorch inference, and standardized postprocessing for comparison.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validate canonical PyTorch vs FP32 ONNX parity.')
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--onnx-model', type=Path, required=True)
    parser.add_argument('--validation-images', type=Path, required=True)
    parser.add_argument('--adapter-script', type=Path, required=True,
                        help='Module defining parity_samples(...) and compare_predictions(...).')
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--sample-count', type=int, default=32)
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location('rrwit_parity_adapter', path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args(); adapter = load_module(args.adapter_script)
    graph = onnx.load(str(args.onnx_model)); onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(args.onnx_model), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    comparisons = []
    for sample in adapter.parity_samples(args.validation_images, args.sample_count):
        tensor = adapter.preprocess(args.model_name, sample)
        pytorch_pred = adapter.pytorch_predict(args.model_name, args.checkpoint, tensor)
        onnx_raw = session.run(None, {input_name: np.asarray(tensor, dtype=np.float32)})
        onnx_pred = adapter.onnx_postprocess(args.model_name, onnx_raw, sample)
        comparisons.append(adapter.compare_predictions(pytorch_pred, onnx_pred))
    result = {'model': args.model_name, 'samples': len(comparisons), 'comparisons': comparisons, 'test_accessed': False}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
