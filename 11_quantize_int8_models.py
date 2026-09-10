#!/usr/bin/env python3
"""Apply the common RRWIT static INT8 QDQ post-training quantization protocol.

Calibration must use a frozen 512-image training-only cohort. Validation and test
images are not used by this script. The common protocol is QDQ + MinMax, signed
8-bit activations and weights, per-tensor quantization, and weighted compute ops
(Conv, Gemm, MatMul) only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import onnx
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

QUANTIZED_OP_TYPES = ['Conv', 'Gemm', 'MatMul']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Static INT8 QDQ quantization for an RRWIT deployment candidate.')
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--fp32-onnx', type=Path, required=True)
    parser.add_argument('--calibration-manifest', type=Path, required=True,
                        help='Text file containing exactly 512 training-image paths.')
    parser.add_argument('--preprocess-script', type=Path, required=True,
                        help='Module defining calibration_reader(model_name, onnx_path, image_paths).')
    parser.add_argument('--output-onnx', type=Path, required=True)
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location('rrwit_quant_preprocess', path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args(); module = load_module(args.preprocess_script)
    images = [Path(line.strip()) for line in args.calibration_manifest.read_text().splitlines() if line.strip()]
    if len(images) != 512:
        raise RuntimeError(f'Expected 512 training calibration images; observed {len(images)}.')
    reader = module.calibration_reader(args.model_name, args.fp32_onnx, images)
    if not isinstance(reader, CalibrationDataReader):
        raise TypeError('calibration_reader() must return an ONNX Runtime CalibrationDataReader.')
    args.output_onnx.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(args.fp32_onnx), model_output=str(args.output_onnx), calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ, calibrate_method=CalibrationMethod.MinMax,
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        per_channel=False, reduce_range=False, op_types_to_quantize=QUANTIZED_OP_TYPES,
    )
    graph = onnx.load(str(args.output_onnx)); onnx.checker.check_model(graph)
    info = {'model': args.model_name, 'calibration_images': 512, 'format': 'QDQ', 'method': 'MinMax',
            'activation_type': 'QInt8', 'weight_type': 'QInt8', 'per_channel': False,
            'quantized_ops': QUANTIZED_OP_TYPES, 'test_accessed': False, 'validation_accessed': False}
    args.output_onnx.with_suffix('.json').write_text(json.dumps(info, indent=2) + '\n')


if __name__ == '__main__':
    main()
