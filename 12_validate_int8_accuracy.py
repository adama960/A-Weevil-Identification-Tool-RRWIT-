#!/usr/bin/env python3
"""Validate frozen INT8 ONNX artifacts against their FP32 validation references.

This stage is validation-only. It does not recalibrate, retune confidence
thresholds, reselect models, or access the locked test split. Project-specific
engineering tolerances are reported as diagnostic qualification flags rather
than external standards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

AP_MAX_DROP = 0.010
F1_MAX_DROP = 0.010


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare FP32 and INT8 validation metrics.')
    parser.add_argument('--fp32-metrics', type=Path, required=True)
    parser.add_argument('--int8-metrics', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fp32 = json.loads(args.fp32_metrics.read_text())
    int8 = json.loads(args.int8_metrics.read_text())
    fields = {'mAP50_95', 'mAP50', 'weighted_f1'}
    if not fields <= fp32.keys() or not fields <= int8.keys():
        raise ValueError(f'Both metric files must contain {sorted(fields)}.')
    drops = {name: float(fp32[name]) - float(int8[name]) for name in fields}
    qualified = drops['mAP50_95'] <= AP_MAX_DROP and drops['mAP50'] <= AP_MAX_DROP and drops['weighted_f1'] <= F1_MAX_DROP
    result = {
        'fp32': {k: float(fp32[k]) for k in fields},
        'int8': {k: float(int8[k]) for k in fields},
        'drops': drops,
        'project_specific_limits': {'mAP50_95': AP_MAX_DROP, 'mAP50': AP_MAX_DROP, 'weighted_f1': F1_MAX_DROP},
        'aggregate_accuracy_qualified': qualified,
        'note': 'Qualification limits are project-specific engineering guards, not universal ONNX Runtime standards.',
        'test_accessed': False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
