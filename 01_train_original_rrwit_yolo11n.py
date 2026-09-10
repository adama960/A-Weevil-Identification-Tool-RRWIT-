#!/usr/bin/env python3
"""Train the original RRWIT model from a COCO-pretrained YOLO11n detector.

This public-facing script reproduces the historical RRWIT training contract described
in the manuscript while avoiding machine-specific filesystem paths.

Workflow
--------
1. Load COCO-pretrained ``yolo11n.pt``.
2. Fine-tune on the RRWIT dataset using the historical 640-pixel training contract.
3. Retain the validation-selected ``best.pt`` checkpoint.
4. Copy the checkpoint to ``RRWIT.v1.0.0.pt``.
5. Optionally evaluate the frozen checkpoint on the test split after training.

The test split is never used for model fitting, early stopping, or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


LOGGER = logging.getLogger("rrwit.train")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the original RRWIT YOLO11n model."
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        required=True,
        help="Path to the RRWIT Ultralytics data.yaml file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which Ultralytics training outputs will be written.",
    )
    parser.add_argument(
        "--pretrained-weights",
        default="yolo11n.pt",
        help="COCO-pretrained YOLO11n weights or local path. Default: yolo11n.pt",
    )
    parser.add_argument(
        "--run-name",
        default="weevil_detector_v6",
        help="Ultralytics run name. Default: weevil_detector_v6",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Training device, e.g. '0' or 'cpu'. Default: first CUDA GPU if available, else CPU.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the frozen best checkpoint on split='test' after training is complete.",
    )
    return parser.parse_args()


def resolve_device(requested: str | None) -> str | int:
    """Resolve the requested compute device to an Ultralytics-compatible value."""
    if requested is not None:
        return requested
    return 0 if torch.cuda.is_available() else "cpu"


def validate_inputs(data_yaml: Path, output_dir: Path) -> None:
    """Validate required inputs and create the output directory."""
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset configuration does not exist: {data_yaml}")
    output_dir.mkdir(parents=True, exist_ok=True)


def write_run_manifest(
    destination: Path,
    args: argparse.Namespace,
    device: str | int,
) -> None:
    """Write the public training configuration before model fitting begins."""
    manifest = {
        "analysis": "Original RRWIT YOLO11n transfer-learning training",
        "data_yaml": str(args.data_yaml),
        "output_dir": str(args.output_dir),
        "pretrained_weights": str(args.pretrained_weights),
        "run_name": args.run_name,
        "device": str(device),
        "seed": 0,
        "deterministic": True,
        "epochs_max": 1000,
        "image_size": 640,
        "batch_size": 16,
        "workers": 6,
        "patience": 15,
        "cache": True,
        "amp": True,
        "optimizer": "auto",
        "archived_optimizer_configuration": {
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
        },
        "loss_weights": {"box": 7.5, "cls": 0.5, "dfl": 1.5},
        "augmentation": {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "translate": 0.1,
            "scale": 0.5,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "close_mosaic": 10,
            "degrees": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "bgr": 0.0,
            "multi_scale": False,
        },
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
    }
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Run RRWIT training and optional frozen test evaluation."""
    args = parse_args()
    validate_inputs(args.data_yaml, args.output_dir)
    device = resolve_device(args.device)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    LOGGER.info("Ultralytics version: %s", ultralytics.__version__)
    LOGGER.info("PyTorch version: %s", torch.__version__)
    LOGGER.info("Training device: %s", device)

    manifest_path = args.output_dir / "rrwit_training_manifest.json"
    write_run_manifest(manifest_path, args, device)
    LOGGER.info("Wrote training manifest: %s", manifest_path)

    model = YOLO(str(args.pretrained_weights))

    # Historical RRWIT training contract. Stochastic augmentation is applied only
    # to the training partition by Ultralytics; validation and test inference are
    # evaluated without stochastic training augmentation.
    training_results = model.train(
        data=str(args.data_yaml),
        epochs=1000,
        imgsz=640,
        batch=16,
        workers=6,
        val=True,
        device=device,
        cache=True,
        project=str(args.output_dir),
        name=args.run_name,
        patience=15,
        seed=0,
        deterministic=True,
        amp=True,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        dropout=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=10,
        multi_scale=False,
        rect=False,
    )

    best_model_path = Path(training_results.save_dir) / "weights" / "best.pt"
    if not best_model_path.is_file():
        raise FileNotFoundError(
            f"Validation-selected checkpoint was not created: {best_model_path}"
        )

    rrwit_checkpoint = best_model_path.with_name("RRWIT.v1.0.0.pt")
    shutil.copy2(best_model_path, rrwit_checkpoint)
    LOGGER.info("Frozen RRWIT checkpoint: %s", rrwit_checkpoint)

    if args.evaluate_test:
        # Test evaluation occurs only after the validation-selected checkpoint has
        # been finalized. It does not influence training or checkpoint selection.
        frozen_model = YOLO(str(rrwit_checkpoint))
        LOGGER.info("Evaluating frozen RRWIT checkpoint on the test split.")
        frozen_model.val(
            data=str(args.data_yaml),
            split="test",
            imgsz=640,
            device=device,
        )
    else:
        LOGGER.info(
            "Test evaluation was not requested. Use --evaluate-test only after "
            "the validation-selected checkpoint has been finalized."
        )


if __name__ == "__main__":
    main()
