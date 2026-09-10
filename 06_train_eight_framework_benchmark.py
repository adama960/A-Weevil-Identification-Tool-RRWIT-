#!/usr/bin/env python3
"""Run one task from the final eight-framework, three-seed RRWIT benchmark.

The final benchmark contains 24 model-seed combinations: eight retained detector
configurations trained independently with seeds 42, 123, and 2025. RT-DETR is
intentionally excluded because it was not retained in the final manuscript benchmark.

This driver consumes a public JSON task configuration and the cleaned RRWIT training
adapter. Dataset, checkpoint, and output paths are caller supplied; the bundled helper module
contains the public framework-neutral and family-specific helper logic.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path
from types import ModuleType

LOGGER = logging.getLogger("rrwit.benchmark_train")
SEEDS = (42, 123, 2025)
MODELS = {
    "yolov5nu": {"input_size": 640, "batch_size": 16},
    "yolov8n": {"input_size": 640, "batch_size": 16},
    "yolo11n_rrwit": {"input_size": 640, "batch_size": 16},
    "yolo12n": {"input_size": 640, "batch_size": 16},
    "yolo26n": {"input_size": 640, "batch_size": 16},
    "ssdlite320_mobilenet_v3_large": {"input_size": 320, "batch_size": 32},
    "efficientdet_d0": {"input_size": 512, "batch_size": 16},
    "fasterrcnn_mobilenet_v3_large_320_fpn": {"input_size": 320, "batch_size": 8},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one retained RRWIT benchmark model/seed task."
    )
    parser.add_argument("--config-json", type=Path, required=True,
                        help="Resolved public task configuration for one model/seed pair.")
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="RRWIT dataset root containing data.yaml and image partitions.")
    parser.add_argument("--coco-root", type=Path, required=True,
                        help="COCO-format RRWIT release used by common evaluation.")
    parser.add_argument(
        "--helper-script",
        type=Path,
        default=Path(__file__).with_name("rrwit_benchmark_helpers.py"),
        help="Consolidated public RRWIT benchmark helper module.",
    )
    parser.add_argument(
        "--adapter-script",
        type=Path,
        default=Path(__file__).with_name("rrwit_eight_framework_training_adapter.py"),
        help="Clean public RRWIT training adapter.",
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Optional override for config['run_dir'].")
    return parser.parse_args()


def load_adapter(path: Path) -> ModuleType:
    """Load the cleaned public adapter and verify its entry point."""
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("rrwit_public_benchmark_adapter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import benchmark adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "train_benchmark_model", None)):
        raise AttributeError("Adapter must define train_benchmark_model(...).")
    return module


def validate_config(config: dict) -> tuple[str, int]:
    """Validate the public task identity against the final benchmark inventory."""
    if "task" not in config:
        raise KeyError("Task configuration must contain a 'task' object.")
    model = str(config["task"].get("model"))
    seed = int(config["task"].get("seed"))
    if model not in MODELS:
        raise ValueError(f"Model is not in the final eight-framework benchmark: {model}")
    if seed not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}; observed {seed}.")
    expected = MODELS[model]
    if int(config["task"].get("image_size")) != expected["input_size"]:
        raise ValueError(f"Input-size mismatch for {model}.")
    if int(config["task"].get("batch_size")) != expected["batch_size"]:
        raise ValueError(f"Batch-size mismatch for {model}.")
    config.setdefault("test_access", "DISABLED")
    config.setdefault("selection_split", "validation")
    return model, seed


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if not args.config_json.is_file():
        raise FileNotFoundError(args.config_json)
    config = json.loads(args.config_json.read_text(encoding="utf-8"))
    model, seed = validate_config(config)

    if args.output_dir is not None:
        config["run_dir"] = str(args.output_dir.resolve())
    if "run_dir" not in config:
        raise KeyError("Task configuration must define 'run_dir' or use --output-dir.")

    # These fields are runtime concerns rather than scientific identifiers.
    config.setdefault("workers", 4)
    config.setdefault("device", "cuda:0")

    adapter = load_adapter(args.adapter_script)
    result = adapter.train_benchmark_model(
        config=config,
        dataset_root=args.dataset_root,
        coco_root=args.coco_root,
        helper_script=args.helper_script,
    )
    LOGGER.info("Completed %s seed %d: %s", model, seed, result)


if __name__ == "__main__":
    main()
