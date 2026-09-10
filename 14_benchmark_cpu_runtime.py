#!/usr/bin/env python3
"""Benchmark frozen FP32 and INT8 ONNX artifacts on CPU under one runtime contract.

The benchmark uses batch size 1, CPUExecutionProvider, four intra-op threads,
one inter-op thread, sequential execution, 16 warm-up images, and three timed
rounds over 128 training-cohort images. Validation/test data are not used.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

RUNTIME_IMAGES = 128
WARMUP_IMAGES = 16
ROUNDS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Controlled RRWIT ONNX CPU runtime benchmark.')
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--onnx-model', type=Path, required=True)
    parser.add_argument('--image-manifest', type=Path, required=True,
                        help='Training-only timing image manifest; at least 128 images required.')
    parser.add_argument('--preprocess-script', type=Path, required=True,
                        help='Module defining preprocess_runtime(model_name, image_path).')
    parser.add_argument('--output-json', type=Path, required=True)
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location('rrwit_runtime_preprocess', path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def session_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


def main() -> None:
    args = parse_args(); module = load_module(args.preprocess_script)
    images = [Path(line.strip()) for line in args.image_manifest.read_text().splitlines() if line.strip()]
    if len(images) < RUNTIME_IMAGES:
        raise RuntimeError(f'At least {RUNTIME_IMAGES} training images are required.')
    images = images[:RUNTIME_IMAGES]
    session = ort.InferenceSession(str(args.onnx_model), sess_options=session_options(), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    tensors = [np.asarray(module.preprocess_runtime(args.model_name, p), dtype=np.float32) for p in images]
    for tensor in tensors[:WARMUP_IMAGES]:
        session.run(None, {input_name: tensor})

    latencies_ms = []
    for _ in range(ROUNDS):
        for tensor in tensors:
            start = time.perf_counter()
            session.run(None, {input_name: tensor})
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    result = {
        'model': args.model_name, 'provider': 'CPUExecutionProvider', 'batch_size': 1,
        'intra_op_threads': 4, 'inter_op_threads': 1,
        'warmup_images': WARMUP_IMAGES, 'timed_rounds': ROUNDS, 'images_per_round': RUNTIME_IMAGES,
        'core_latency_ms_mean': statistics.mean(latencies_ms),
        'core_latency_ms_median': statistics.median(latencies_ms),
        'core_latency_ms_p95': float(np.quantile(latencies_ms, 0.95)),
        'throughput_images_per_second': 1000.0 / statistics.mean(latencies_ms),
        'validation_test_accessed': False,
        'note': 'This measures ONNX Runtime model-core latency after preprocessing, not full application latency.',
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
