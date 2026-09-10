#!/usr/bin/env python3
"""Generate the predeclared EfficientDet-D0 INT8 diagnostic quantization matrix.

The authoritative common-protocol result is S8S8 per-tensor. This diagnostic
branch additionally evaluates S8S8 per-channel, U8U8 per-tensor, and U8U8
per-channel variants to understand architecture-specific sensitivity. All
artifacts must be generated from the same frozen training-only calibration
cohort before any validation comparison is performed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

VARIANTS = {
    's8s8_per_tensor': (QuantType.QInt8, QuantType.QInt8, False),
    's8s8_per_channel': (QuantType.QInt8, QuantType.QInt8, True),
    'u8u8_per_tensor': (QuantType.QUInt8, QuantType.QUInt8, False),
    'u8u8_per_channel': (QuantType.QUInt8, QuantType.QUInt8, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='EfficientDet-D0 INT8 diagnostic quantization matrix.')
    parser.add_argument('--fp32-onnx', type=Path, required=True)
    parser.add_argument('--calibration-manifest', type=Path, required=True)
    parser.add_argument('--preprocess-script', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location('rrwit_effdet_quant_preprocess', path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    module = load_module(args.preprocess_script)
    images = [Path(line.strip()) for line in args.calibration_manifest.read_text().splitlines() if line.strip()]
    if len(images) != 512:
        raise RuntimeError(f'Expected 512 training-only calibration images; observed {len(images)}.')
    records = []
    for name, (activation, weight, per_channel) in VARIANTS.items():
        reader = module.calibration_reader('efficientdet_d0', args.fp32_onnx, images)
        if not isinstance(reader, CalibrationDataReader):
            raise TypeError('calibration_reader() must return CalibrationDataReader.')
        output = args.output_dir / f'EfficientDetD0_{name}.onnx'
        quantize_static(
            str(args.fp32_onnx), str(output), reader,
            quant_format=QuantFormat.QDQ, calibrate_method=CalibrationMethod.MinMax,
            activation_type=activation, weight_type=weight,
            per_channel=per_channel, reduce_range=False,
            op_types_to_quantize=['Conv', 'Gemm', 'MatMul'],
        )
        records.append({'variant': name, 'activation_type': str(activation), 'weight_type': str(weight),
                        'per_channel': per_channel, 'artifact': str(output), 'bytes': output.stat().st_size})
    (args.output_dir / 'EfficientDetD0_INT8_diagnostic_manifest.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
