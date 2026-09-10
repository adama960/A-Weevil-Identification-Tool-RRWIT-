#!/usr/bin/env python3
"""Framework adapter for the final eight-model RRWIT benchmark.

This public module retains the scientific training/evaluation logic used by the
final RRWIT benchmark while removing project-specific filesystem paths, Slurm
controls, release-freeze machinery, hash-pinned internal infrastructure, and the
RT-DETR branch that was not retained in the final manuscript benchmark.

The benchmark contains exactly eight detector configurations:
    YOLOv5nu, YOLOv8n, RRWIT/YOLO11n, YOLO12n, YOLO26n,
    SSDLite320 MobileNetV3-Large, EfficientDet-D0, and
    Faster R-CNN MobileNetV3-Large-320 FPN.

Model construction, canonical dataset adapters, pretrained-weight transfer, and
optimizer construction are provided by the consolidated public
``rrwit_benchmark_helpers.py`` module. That module was derived from the final
validated helper logic while excluding internal preflight and audit machinery.

The held-out test set is never opened by this training adapter. Validation data
are used for epoch-level monitoring and checkpoint selection, and all detector
predictions are normalized through a common COCO evaluation boundary.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from effdet.data import resolve_input_config as resolve_effdet_input_config
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

SUPPORTED_MODELS = {
    "yolov5nu",
    "yolov8n",
    "yolo11n_rrwit",
    "yolo12n",
    "yolo26n",
    "ssdlite320_mobilenet_v3_large",
    "efficientdet_d0",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
}
YOLO_MODELS = {
    "yolov5nu", "yolov8n", "yolo11n_rrwit", "yolo12n", "yolo26n"
}
TORCHVISION_MODELS = {
    "ssdlite320_mobilenet_v3_large",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
}
STANDARD_ARTIFACTS = [
    "resolved_config.json",
    "training_history.csv",
    "best_checkpoint.pt",
    "last_checkpoint.pt",
    "validation_predictions_coco.json",
    "validation_overall_metrics.json",
    "validation_per_class_metrics.csv",
    "resource_usage.json",
    "prediction_filtering.json",
]

_DATASET_ROOT: Path | None = None
_COCO_ROOT: Path | None = None
_HELPERS: Any = None


def _load_module(path: Path, module_name: str) -> Any:
    """Import one public helper module from an explicit path."""
    if not path.is_file():
        raise FileNotFoundError(f"Required helper module is missing: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_runtime(
    dataset_root: Path,
    coco_root: Path,
    helper_script: Path,
) -> None:
    """Configure public paths and load the consolidated benchmark helper module."""
    global _DATASET_ROOT, _COCO_ROOT, _HELPERS

    _DATASET_ROOT = dataset_root.resolve()
    _COCO_ROOT = coco_root.resolve()
    if not _DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {_DATASET_ROOT}")
    if not _COCO_ROOT.is_dir():
        raise FileNotFoundError(f"COCO root does not exist: {_COCO_ROOT}")

    _HELPERS = _load_module(helper_script, "rrwit_public_benchmark_helpers")


def _require_runtime() -> tuple[Path, Path, Any, Any, Any]:
    """Return configured roots and the helper module in legacy-compatible positions."""
    if _DATASET_ROOT is None or _COCO_ROOT is None or _HELPERS is None:
        raise RuntimeError("configure_runtime(...) must be called before training.")
    # Existing family-specific functions use separate logical aliases for loader,
    # architecture and optimizer responsibilities. They intentionally point to
    # the same consolidated public helper module.
    return _DATASET_ROOT, _COCO_ROOT, _HELPERS, _HELPERS, _HELPERS

def atomic_json(path: Path, value: Any) -> None:
    """Write JSON atomically to avoid incomplete metadata files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a non-empty list of dictionaries as CSV."""
    if not rows:
        raise RuntimeError(f"Refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_worker(_: int) -> None:
    """Derive NumPy/Python dataloader-worker seeds from the PyTorch worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class RandomHorizontalFlip:
    """Apply the matched non-YOLO image/box horizontal flip with probability 0.5."""

    def __init__(self, loader: Any, probability: float = 0.5):
        self.loader = loader
        self.probability = probability

    def __call__(self, record: dict[str, Any]) -> dict[str, Any]:
        output = self.loader.clone_record(record)
        if torch.rand(()) >= self.probability:
            return output
        output["image"] = torch.flip(output["image"], dims=[2])
        original = output["target"]["boxes"].clone()
        width = output["width"]
        output["target"]["boxes"][:, 0] = width - original[:, 2]
        output["target"]["boxes"][:, 2] = width - original[:, 0]
        self.loader.validate_canonical(output)
        return output


def make_loaders(batch_size: int, workers: int, seed: int):
    """Build deterministic train/validation dataloaders from the frozen COCO release."""
    dataset_root, coco_root, loader, _, _ = _require_runtime()
    train_json = coco_root / "annotations" / "instances_train.json"
    val_json = coco_root / "annotations" / "instances_val.json"
    for path in (train_json, val_json):
        if not path.is_file():
            raise FileNotFoundError(path)

    train = loader.FrozenCocoDataset(
        dataset_root, train_json, RandomHorizontalFlip(loader)
    )
    validation = loader.FrozenCocoDataset(
        dataset_root, val_json, loader.eval_identity
    )

    # Frozen RRWIT benchmark census reported in the manuscript.
    if len(train) != 13_510 or len(validation) != 1_690:
        raise RuntimeError(
            f"Dataset census differs from the frozen benchmark: "
            f"train={len(train)}, validation={len(validation)}"
        )

    generator = torch.Generator().manual_seed(seed)
    common = dict(
        num_workers=workers,
        collate_fn=loader.collate_records,
        worker_init_fn=seed_worker,
        persistent_workers=False,
        pin_memory=True,
    )
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    train_loader.rrwit_generator = generator
    validation_loader = DataLoader(
        validation,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, validation_loader, validation.coco


def coco_metrics(coco_gt: Any, predictions: list[dict[str, Any]]):
    """Calculate common COCO detection metrics and per-class AP50-95."""
    if not predictions:
        per_class = []
        for category_id in range(1, 15):
            category = coco_gt.loadCats([category_id])[0]
            per_class.append({
                "category_id": category_id,
                "category": category["name"],
                "AP50_95": 0.0,
                "validation_support": len(coco_gt.getAnnIds(catIds=[category_id])),
            })
        return {"AP50_95": 0.0, "AP50": 0.0, "AP75": 0.0, "AR100": 0.0}, per_class

    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = sorted(coco_gt.getImgIds())
    evaluator.params.catIds = list(range(1, 15))
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metrics = {
        "AP50_95": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "AR100": float(evaluator.stats[8]),
    }
    precision = evaluator.eval["precision"]
    per_class = []
    for index, category_id in enumerate(evaluator.params.catIds):
        values = precision[:, :, index, 0, 2]
        values = values[values > -1]
        category = coco_gt.loadCats([category_id])[0]
        per_class.append({
            "category_id": category_id,
            "category": category["name"],
            "AP50_95": float(values.mean()) if values.size else float("nan"),
            "validation_support": len(coco_gt.getAnnIds(catIds=[category_id])),
        })
    return metrics, per_class


def empty_prediction_filtering() -> dict[str, int]:
    """Return framework-neutral prediction-canonicalization counters."""
    return {
        "raw_predictions": 0,
        "nonfinite_rejected": 0,
        "invalid_label_rejected": 0,
        "invalid_score_rejected": 0,
        "below_score_floor_rejected": 0,
        "boxes_clipped": 0,
        "nonpositive_after_clipping_rejected": 0,
        "pre_maxdet_accepted_predictions": 0,
        "maxdet_truncated": 0,
        "accepted_predictions": 0,
        "images_with_predictions": 0,
        "maximum_predictions_per_image": 0,
    }


def merge_prediction_filtering(total: dict[str, int], current: dict[str, int]) -> None:
    """Merge one image's filtering evidence into a run-level summary."""
    for key, value in current.items():
        if key == "maximum_predictions_per_image":
            total[key] = max(total[key], value)
        else:
            total[key] += value


def canonicalize_image_predictions(
    rows: Any,
    record: dict[str, Any],
    score_floor: float = 0.001,
    max_detections: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Normalize one detector's XYXY rows into valid COCO result dictionaries."""
    diagnostics = empty_prediction_filtering()
    accepted: list[dict[str, Any]] = []
    width, height = float(record["width"]), float(record["height"])
    image_id = int(record["target"]["image_id"])

    for row in rows:
        diagnostics["raw_predictions"] += 1
        x1, y1, x2, y2, score, label = (float(value) for value in row)
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2, score, label)):
            diagnostics["nonfinite_rejected"] += 1
            continue
        category = int(label)
        if label != category or category not in range(1, 15):
            diagnostics["invalid_label_rejected"] += 1
            continue
        if not 0.0 <= score <= 1.0:
            diagnostics["invalid_score_rejected"] += 1
            continue
        if score < score_floor:
            diagnostics["below_score_floor_rejected"] += 1
            continue

        original = (x1, y1, x2, y2)
        x1, x2 = min(max(x1, 0.0), width), min(max(x2, 0.0), width)
        y1, y2 = min(max(y1, 0.0), height), min(max(y2, 0.0), height)
        if (x1, y1, x2, y2) != original:
            diagnostics["boxes_clipped"] += 1
        if x2 <= x1 or y2 <= y1:
            diagnostics["nonpositive_after_clipping_rejected"] += 1
            continue

        accepted.append({
            "image_id": image_id,
            "category_id": category,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": score,
        })

    accepted.sort(key=lambda item: item["score"], reverse=True)
    diagnostics["pre_maxdet_accepted_predictions"] = len(accepted)
    diagnostics["maxdet_truncated"] = max(0, len(accepted) - max_detections)
    accepted = accepted[:max_detections]
    diagnostics["accepted_predictions"] = len(accepted)
    diagnostics["images_with_predictions"] = int(bool(accepted))
    diagnostics["maximum_predictions_per_image"] = len(accepted)
    return accepted, diagnostics


def _build_optimizer(model: torch.nn.Module, model_name: str, recipe: dict[str, Any]):
    """Delegate the frozen optimizer construction to the public optimizer helper."""
    _, _, _, _, optimizer_helper = _require_runtime()
    optimizer_helper.ACTIVE_MODEL = model_name
    optimizer_helper.ACTIVE_RECIPE = recipe
    return optimizer_helper.build_optimizer(model)[0]


def _build_plateau(recipe: dict[str, Any], optimizer: torch.optim.Optimizer):
    spec = recipe["scheduler"]
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(spec["factor"]),
        patience=int(spec["patience_validations"]),
        threshold=float(spec["threshold"]),
        threshold_mode=spec["threshold_mode"],
        cooldown=int(spec["cooldown"]),
        min_lr=float(spec["min_lr"]),
    )


def _stopping_policy(config: dict[str, Any], model_name: str) -> tuple[int, int, int]:
    """Resolve maximum epochs, minimum epochs, and validation patience."""
    common = config["common"]
    family = (
        "ultralytics" if model_name in YOLO_MODELS
        else "torchvision" if model_name in TORCHVISION_MODELS
        else "efficientdet"
    )
    configured_max = common["maximum_epochs"]
    maximum = int(
        config.get("max_epochs_override")
        or (configured_max[family] if isinstance(configured_max, dict) else configured_max)
    )
    configured_min = common.get("minimum_epochs", {})
    minimum = int(
        configured_min.get(family, 0)
        if isinstance(configured_min, dict)
        else configured_min
    )
    patience = int(common["early_stopping"]["patience_validations"])
    if not (1 <= maximum and 0 <= minimum <= maximum and patience >= 1):
        raise RuntimeError(
            f"Invalid stopping policy: max={maximum}, min={minimum}, patience={patience}"
        )
    return maximum, minimum, patience


def _make_scaler(config: dict[str, Any]) -> torch.amp.GradScaler:
    """Build AMP scaler from the frozen task's recorded initial scale."""
    initial_scale = float(config["amp_initial_scale"])
    if not math.isfinite(initial_scale) or initial_scale <= 0:
        raise RuntimeError(f"Invalid AMP initial scale: {initial_scale}")
    return torch.amp.GradScaler(
        "cuda",
        enabled=bool(config["common"]["amp"]),
        init_scale=initial_scale,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
    )


def _amp_update(
    loss: torch.Tensor,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    clipping: dict[str, Any] | None,
) -> bool:
    """Execute one numerically guarded AMP optimizer update."""
    if not torch.isfinite(loss):
        raise RuntimeError(f"Nonfinite training loss: {float(loss.detach())}")
    old_scale = float(scaler.get_scale())
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients:
        raise RuntimeError("Backward pass produced no trainable gradients.")
    finite = all(torch.isfinite(g).all().item() for g in gradients)
    if finite and clipping:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(clipping["max_norm"]),
            norm_type=float(clipping["norm_type"]),
            error_if_nonfinite=True,
        )
    scaler.step(optimizer)
    scaler.update()
    if not finite and float(scaler.get_scale()) >= old_scale:
        raise RuntimeError("GradScaler failed to back off after nonfinite gradients.")
    return finite


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best: float,
    best_epoch: int,
    patience: int,
    optimizer_steps: int,
    history: list[dict[str, Any]],
    skipped_amp_updates: int,
    config: dict[str, Any],
    generator_state: torch.Tensor,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_validation_coco_ap50_95": best,
        "best_epoch": best_epoch,
        "patience_counter": patience,
        "optimizer_steps": optimizer_steps,
        "history": history,
        "skipped_amp_updates": skipped_amp_updates,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "data_generator_rng": generator_state.clone(),
        "config": config,
    }


def save_common_artifacts(
    run_dir: Path,
    config: dict[str, Any],
    history: list[dict[str, Any]],
    metrics: dict[str, Any],
    per_class: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    resource: dict[str, Any],
) -> None:
    """Write common scientific outputs used by downstream evaluation."""
    atomic_json(run_dir / "resolved_config.json", config)
    write_csv(run_dir / "training_history.csv", history)
    atomic_json(run_dir / "validation_predictions_coco.json", predictions)
    atomic_json(run_dir / "validation_overall_metrics.json", metrics)
    write_csv(run_dir / "validation_per_class_metrics.csv", per_class)
    atomic_json(run_dir / "resource_usage.json", resource)


@torch.inference_mode()
def evaluate_torchvision(model: torch.nn.Module, validation, device, coco_gt):
    """Evaluate an SSDLite/Faster R-CNN checkpoint through the common COCO boundary."""
    model.eval()
    predictions: list[dict[str, Any]] = []
    filtering = empty_prediction_filtering()
    for _, _, records in validation:
        outputs = model([item["image"].to(device, non_blocking=True) for item in records])
        for record, output in zip(records, outputs):
            rows = [
                (*box.cpu().tolist(), float(score), int(label))
                for box, score, label in zip(
                    output["boxes"], output["scores"], output["labels"]
                )
            ]
            accepted, evidence = canonicalize_image_predictions(rows, record)
            predictions.extend(accepted)
            merge_prediction_filtering(filtering, evidence)
    metrics, per_class = coco_metrics(coco_gt, predictions)
    return metrics, per_class, predictions, filtering


def train_torchvision(config: dict[str, Any]) -> dict[str, Any]:
    """Train SSDLite320 or Faster R-CNN under the frozen RRWIT task contract."""
    _, _, loader, architectures, _ = _require_runtime()
    task, recipe, common = config["task"], config["recipe"], config["common"]
    name = str(task["model"])
    run_dir = Path(config["run_dir"])
    device = torch.device(config["device"])
    train_loader, validation, coco_gt = make_loaders(
        int(task["batch_size"]), int(config["workers"]), int(task["seed"])
    )

    constructor = (
        architectures.ssdlite320_mobilenet_v3_large
        if name.startswith("ssdlite")
        else architectures.fasterrcnn_mobilenet_v3_large_320_fpn
    )
    model = constructor(weights=None, weights_backbone=None,
                        num_classes=architectures.NUM_CLASSES + 1)
    checkpoint = torch.load(Path(task["checkpoint"]), map_location="cpu", weights_only=True)
    transfer = architectures.compatible_transfer(model, checkpoint)
    model.to(device)

    optimizer = _build_optimizer(model, name, recipe)
    scheduler = _build_plateau(recipe, optimizer)
    scaler = _make_scaler(config)
    warm_steps = int(recipe["warmup"]["optimizer_steps"])
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if warm_steps:
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr / warm_steps

    max_epochs, minimum_epochs, patience_limit = _stopping_policy(config, name)
    best = -math.inf
    best_epoch = optimizer_steps = patience = skipped = 0
    history: list[dict[str, Any]] = []
    final = None
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(max_epochs):
        model.train()
        loss_sum = 0.0
        batches = 0
        for _, _, records in train_loader:
            optimizer.zero_grad(set_to_none=True)
            images = [record["image"].to(device, non_blocking=True) for record in records]
            targets = [
                {
                    "boxes": loader.make_adapters(record)["torchvision"]["boxes"].float().to(device),
                    "labels": loader.make_adapters(record)["torchvision"]["labels"].long().to(device),
                }
                for record in records
            ]
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(common["amp"])):
                components = model(images, targets)
                loss = sum(components.values())
            if _amp_update(loss, model, optimizer, scaler, recipe.get("gradient_clipping")):
                optimizer_steps += 1
                if warm_steps and optimizer_steps <= warm_steps:
                    for group, base_lr in zip(optimizer.param_groups, base_lrs):
                        group["lr"] = base_lr * optimizer_steps / warm_steps
            else:
                skipped += 1
            loss_sum += float(loss.detach())
            batches += 1

        metrics, per_class, predictions, filtering = evaluate_torchvision(
            model, validation, device, coco_gt
        )
        if optimizer_steps >= warm_steps:
            scheduler.step(metrics["AP50_95"])
        improved = metrics["AP50_95"] > best
        if improved:
            best, best_epoch, patience = metrics["AP50_95"], epoch + 1, 0
            final = (metrics, per_class, predictions, filtering)
        else:
            patience += 1

        row = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(batches, 1),
            **metrics,
            "optimizer_steps": optimizer_steps,
            "skipped_amp_updates": skipped,
            "scaler_scale": float(scaler.get_scale()),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        write_csv(run_dir / "training_history.csv", history)
        payload = _checkpoint_payload(
            model, optimizer, scheduler, scaler, epoch, best, best_epoch,
            patience, optimizer_steps, history, skipped, config,
            train_loader.rrwit_generator.get_state(),
        )
        torch.save(payload, run_dir / "last_checkpoint.pt")
        if improved:
            torch.save(payload, run_dir / "best_checkpoint.pt")
        if epoch + 1 >= minimum_epochs and patience >= patience_limit:
            break

    if final is None:
        raise RuntimeError(f"{name}: no validation-selected checkpoint was produced.")
    metrics, per_class, predictions, filtering = final
    atomic_json(run_dir / "prediction_filtering.json", filtering)
    save_common_artifacts(
        run_dir, config, history, metrics, per_class, predictions,
        {
            "wall_seconds": time.monotonic() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "transfer": transfer,
            "skipped_amp_updates": skipped,
        },
    )
    return {
        "status": "TRAINING_PASS",
        "best_epoch": best_epoch,
        "best_validation_coco_ap50_95": best,
        "epochs_completed": len(history),
        "optimizer_steps": optimizer_steps,
        "artifacts": STANDARD_ARTIFACTS,
    }


def yolo_validation_predictions(model_path: Path, validation, task: dict[str, Any]):
    """Evaluate a YOLO checkpoint through the same canonical COCO boundary."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    predictions: list[dict[str, Any]] = []
    filtering = empty_prediction_filtering()
    for _, _, records in validation:
        results = model.predict(
            source=[record["image_path"] for record in records],
            imgsz=int(task["image_size"]),
            conf=0.001,
            iou=0.70,
            max_det=100,
            device=0,
            verbose=False,
            stream=False,
        )
        for record, result in zip(records, results):
            rows = [] if result.boxes is None else [
                (*box.cpu().tolist(), float(score), int(label) + 1)
                for box, score, label in zip(
                    result.boxes.xyxy, result.boxes.conf, result.boxes.cls
                )
            ]
            accepted, evidence = canonicalize_image_predictions(rows, record)
            predictions.extend(accepted)
            merge_prediction_filtering(filtering, evidence)
    return predictions, filtering


def train_yolo(config: dict[str, Any]) -> dict[str, Any]:
    """Train one retained Ultralytics YOLO configuration and select by COCO AP50-95."""
    from ultralytics import YOLO

    dataset_root, _, _, _, _ = _require_runtime()
    task, recipe, common = config["task"], config["recipe"], config["common"]
    name = str(task["model"])
    run_dir = Path(config["run_dir"])
    native_dir = run_dir / "native_ultralytics"
    native_dir.mkdir(parents=True, exist_ok=True)
    seed = int(task["seed"])
    batch = int(task["batch_size"])
    max_epochs, minimum_epochs, patience_limit = _stopping_policy(config, name)

    source_yaml = dataset_root / "data.yaml"
    if not source_yaml.is_file():
        raise FileNotFoundError(source_yaml)
    data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    data.pop("test", None)
    data["path"] = str(dataset_root)
    train_val_yaml = run_dir / "train_val_only.yaml"
    train_val_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    _, validation, coco_gt = make_loaders(batch, int(config["workers"]), seed)
    model = YOLO(str(task["checkpoint"]))
    canonical = {
        "best": -math.inf,
        "best_epoch": 0,
        "patience": 0,
        "history": [],
        "metrics": None,
        "per_class": None,
        "predictions": None,
        "filtering": None,
    }

    def canonical_epoch_selection(trainer):
        trainer.save_model()
        epoch = int(trainer.epoch) + 1
        predictions, filtering = yolo_validation_predictions(
            Path(trainer.last), validation, task
        )
        metrics, per_class = coco_metrics(coco_gt, predictions)
        value = float(metrics["AP50_95"])
        if not math.isfinite(value):
            raise RuntimeError(f"{name}: nonfinite validation AP50-95 at epoch {epoch}")
        improved = value > float(canonical["best"])
        if improved:
            canonical.update({
                "best": value,
                "best_epoch": epoch,
                "patience": 0,
                "metrics": metrics,
                "per_class": per_class,
                "predictions": predictions,
                "filtering": filtering,
            })
            shutil.copy2(trainer.last, run_dir / "best_checkpoint.pt")
            atomic_json(run_dir / "validation_predictions_coco.json", predictions)
            atomic_json(run_dir / "validation_overall_metrics.json", metrics)
            write_csv(run_dir / "validation_per_class_metrics.csv", per_class)
            atomic_json(run_dir / "prediction_filtering.json", filtering)
        else:
            canonical["patience"] += 1

        canonical["history"].append({
            "epoch": epoch,
            **metrics,
            "canonical_improved": improved,
            "canonical_patience_counter": int(canonical["patience"]),
        })
        write_csv(run_dir / "training_history.csv", canonical["history"])
        trainer.stop = bool(
            epoch >= minimum_epochs and canonical["patience"] >= patience_limit
        )

    model.add_callback("on_fit_epoch_end", canonical_epoch_selection)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    model.train(
        data=str(train_val_yaml),
        epochs=max_epochs,
        patience=0,
        batch=batch,
        imgsz=int(task["image_size"]),
        workers=int(config["workers"]),
        device=0,
        project=str(native_dir),
        name="train",
        exist_ok=False,
        pretrained=True,
        optimizer="SGD",
        lr0=float(recipe["optimizer"]["lr0"]),
        lrf=float(recipe["scheduler"]["final_lr_fraction"]),
        momentum=float(recipe["optimizer"]["momentum"]),
        weight_decay=float(recipe["optimizer"]["weight_decay"]),
        nbs=int(recipe["optimizer"]["nbs"]),
        cos_lr=False,
        warmup_epochs=float(recipe["warmup"]["epochs"]),
        warmup_momentum=float(recipe["warmup"]["initial_momentum"]),
        warmup_bias_lr=float(recipe["warmup"]["initial_bias_lr"]),
        box=float(recipe["loss_weights"]["box"]),
        cls=float(recipe["loss_weights"]["cls"]),
        dfl=float(recipe["loss_weights"]["dfl"]),
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
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        cache=False,
        amp=True,
        val=True,
        save=True,
        save_period=-1,
        plots=False,
        seed=seed,
        deterministic=True,
        conf=0.001,
        iou=0.70,
        max_det=100,
        verbose=True,
    )

    train_dir = native_dir / "train"
    last_source = train_dir / "weights" / "last.pt"
    if not (run_dir / "best_checkpoint.pt").is_file() or not last_source.is_file():
        raise RuntimeError(f"{name}: expected checkpoints were not created.")
    shutil.copy2(last_source, run_dir / "last_checkpoint.pt")

    metrics = canonical["metrics"]
    predictions = canonical["predictions"]
    filtering = canonical["filtering"]
    per_class = canonical["per_class"]
    if metrics is None or predictions is None or filtering is None or per_class is None:
        raise RuntimeError(f"{name}: canonical validation evidence is incomplete.")

    save_common_artifacts(
        run_dir,
        config,
        canonical["history"],
        metrics,
        per_class,
        predictions,
        {
            "wall_seconds": time.monotonic() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "canonical_selection": True,
            "physical_batch_size": batch,
            "nbs": int(recipe["optimizer"]["nbs"]),
        },
    )
    return {
        "status": "TRAINING_PASS",
        "best_epoch": int(canonical["best_epoch"]),
        "best_validation_coco_ap50_95": float(metrics["AP50_95"]),
        "epochs_completed": len(canonical["history"]),
        "artifacts": STANDARD_ARTIFACTS,
    }


def effdet_data_config(bench: torch.nn.Module) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Resolve normalization through the EfficientDet model's public data config."""
    config = getattr(getattr(bench, "model", bench), "config", None)
    if config is None:
        raise RuntimeError("EfficientDet model does not expose its input configuration.")
    input_config = resolve_effdet_input_config({}, model_config=config)
    mean = tuple(float(value) for value in input_config["mean"])
    std = tuple(float(value) for value in input_config["std"])
    return mean, std


def effdet_letterbox(loader, records, size, device, mean, std):
    """Create normalized top-left EfficientDet letterboxes and absolute yxyx targets."""
    images, adapters, scales, original_sizes = [], [], [], []
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    for record in records:
        scale = min(size / record["width"], size / record["height"])
        new_w = max(1, int(record["width"] * scale))
        new_h = max(1, int(record["height"] * scale))
        resized = F.interpolate(
            record["image"].unsqueeze(0),
            (new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )[0]
        canvas = mean_tensor.expand(3, size, size).clone().to(dtype=resized.dtype)
        canvas[:, :new_h, :new_w] = resized
        canvas = (canvas - mean_tensor.to(canvas.dtype)) / std_tensor.to(canvas.dtype)
        adapter = loader.make_adapters(record)["efficientdet"]
        adapters.append({
            "boxes": adapter["boxes"].float() * scale,
            "labels": adapter["labels"].long(),
        })
        images.append(canvas)
        scales.append(scale)
        original_sizes.append([int(record["height"]), int(record["width"])])

    maximum = max(len(item["labels"]) for item in adapters)
    boxes = torch.full((len(records), maximum, 4), -1.0, device=device)
    labels = torch.full((len(records), maximum), -1, dtype=torch.int64, device=device)
    for index, item in enumerate(adapters):
        count = len(item["labels"])
        boxes[index, :count] = item["boxes"].to(device)
        labels[index, :count] = item["labels"].to(device)
    target = {
        "bbox": boxes,
        "cls": labels,
        "img_size": torch.tensor(original_sizes, dtype=torch.float32, device=device),
        "img_scale": torch.tensor([1 / value for value in scales], dtype=torch.float32, device=device),
    }
    return torch.stack(images).to(device, non_blocking=True), target


@torch.inference_mode()
def evaluate_effdet(bench, validation, loader, device, coco_gt, size, mean, std):
    """Evaluate EfficientDet through the shared COCO result boundary."""
    bench.eval()
    predictions: list[dict[str, Any]] = []
    filtering = empty_prediction_filtering()
    for _, _, records in validation:
        images, info = effdet_letterbox(loader, records, size, device, mean, std)
        detections = bench(images, info)
        for record, result in zip(records, detections):
            rows = [tuple(float(value) for value in row[:6].cpu()) for row in result]
            accepted, evidence = canonicalize_image_predictions(rows, record)
            predictions.extend(accepted)
            merge_prediction_filtering(filtering, evidence)
    metrics, per_class = coco_metrics(coco_gt, predictions)
    return metrics, per_class, predictions, filtering


def train_effdet(config: dict[str, Any]) -> dict[str, Any]:
    """Train EfficientDet-D0 with the family-specific 512-pixel RRWIT configuration."""
    _, _, loader, architectures, _ = _require_runtime()
    task, recipe, common = config["task"], config["recipe"], config["common"]
    run_dir = Path(config["run_dir"])
    device = torch.device(config["device"])
    size = int(task["image_size"])
    train_loader, validation, coco_gt = make_loaders(
        int(task["batch_size"]), int(config["workers"]), int(task["seed"])
    )

    bench = architectures.create_model(
        "tf_efficientdet_d0",
        bench_task="train",
        num_classes=architectures.NUM_CLASSES,
        pretrained=False,
        pretrained_backbone=False,
        bench_labeler=True,
    )
    checkpoint_state = torch.load(Path(task["checkpoint"]), map_location="cpu", weights_only=True)
    transfer = architectures.compatible_transfer(bench.model, checkpoint_state)
    mean, std = effdet_data_config(bench)
    bench.to(device)

    optimizer = _build_optimizer(bench, "efficientdet_d0", recipe)
    scheduler = _build_plateau(recipe, optimizer)
    scaler = _make_scaler(config)
    warm_steps = int(recipe["warmup"]["optimizer_steps"])
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if warm_steps:
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr / warm_steps

    max_epochs, minimum_epochs, patience_limit = _stopping_policy(config, "efficientdet_d0")
    best = -math.inf
    best_epoch = optimizer_steps = patience = skipped = 0
    history: list[dict[str, Any]] = []
    final = None
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(max_epochs):
        bench.train()
        loss_sum = class_sum = box_sum = 0.0
        batches = 0
        for _, _, records in train_loader:
            optimizer.zero_grad(set_to_none=True)
            images, targets = effdet_letterbox(loader, records, size, device, mean, std)
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(common["amp"])):
                output = bench(images, targets)
                loss = output["loss"]
            if _amp_update(loss, bench, optimizer, scaler, recipe.get("gradient_clipping")):
                optimizer_steps += 1
                if warm_steps and optimizer_steps <= warm_steps:
                    for group, base_lr in zip(optimizer.param_groups, base_lrs):
                        group["lr"] = base_lr * optimizer_steps / warm_steps
            else:
                skipped += 1
            loss_sum += float(loss.detach())
            class_sum += float(output["class_loss"].detach())
            box_sum += float(output["box_loss"].detach())
            batches += 1

        predictor = architectures.create_model(
            "tf_efficientdet_d0",
            bench_task="predict",
            num_classes=architectures.NUM_CLASSES,
            pretrained=False,
            pretrained_backbone=False,
        ).to(device)
        predictor.model.load_state_dict(bench.model.state_dict(), strict=True)
        metrics, per_class, predictions, filtering = evaluate_effdet(
            predictor, validation, loader, device, coco_gt, size, mean, std
        )
        del predictor

        if optimizer_steps >= warm_steps:
            scheduler.step(metrics["AP50_95"])
        improved = metrics["AP50_95"] > best
        if improved:
            best, best_epoch, patience = metrics["AP50_95"], epoch + 1, 0
            final = (metrics, per_class, predictions, filtering)
        else:
            patience += 1

        row = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(batches, 1),
            "train_class_loss": class_sum / max(batches, 1),
            "train_box_loss": box_sum / max(batches, 1),
            **metrics,
            "optimizer_steps": optimizer_steps,
            "skipped_amp_updates": skipped,
            "scaler_scale": float(scaler.get_scale()),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        write_csv(run_dir / "training_history.csv", history)
        payload = _checkpoint_payload(
            bench, optimizer, scheduler, scaler, epoch, best, best_epoch,
            patience, optimizer_steps, history, skipped, config,
            train_loader.rrwit_generator.get_state(),
        )
        torch.save(payload, run_dir / "last_checkpoint.pt")
        if improved:
            torch.save(payload, run_dir / "best_checkpoint.pt")
        if epoch + 1 >= minimum_epochs and patience >= patience_limit:
            break

    if final is None:
        raise RuntimeError("EfficientDet-D0 did not produce a validation-selected checkpoint.")
    metrics, per_class, predictions, filtering = final
    atomic_json(run_dir / "prediction_filtering.json", filtering)
    save_common_artifacts(
        run_dir, config, history, metrics, per_class, predictions,
        {
            "wall_seconds": time.monotonic() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "transfer": transfer,
            "skipped_amp_updates": skipped,
            "resize_policy": "aspect-preserving top-left mean-color letterbox to 512",
            "normalization": {"mean": mean, "std": std},
        },
    )
    return {
        "status": "TRAINING_PASS",
        "best_epoch": best_epoch,
        "best_validation_coco_ap50_95": best,
        "epochs_completed": len(history),
        "optimizer_steps": optimizer_steps,
        "artifacts": STANDARD_ARTIFACTS,
    }


def train_task(config: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one retained eight-framework benchmark task."""
    model_name = str(config["task"]["model"])
    if model_name not in SUPPORTED_MODELS:
        raise RuntimeError(f"Unsupported final-benchmark model: {model_name}")
    if config.get("test_access", "DISABLED") != "DISABLED":
        raise RuntimeError("Training adapter requires test_access='DISABLED'.")
    if config.get("selection_split", "validation") != "validation":
        raise RuntimeError("Checkpoint selection must use validation data only.")

    seed = int(config["task"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if model_name in YOLO_MODELS:
        return train_yolo(config)
    if model_name in TORCHVISION_MODELS:
        return train_torchvision(config)
    if model_name == "efficientdet_d0":
        return train_effdet(config)
    raise AssertionError("Unreachable model dispatch branch.")


def train_benchmark_model(
    *,
    config: dict[str, Any],
    dataset_root: Path,
    coco_root: Path,
    helper_script: Path,
) -> dict[str, Any]:
    """Public entry point used by ``06_train_eight_framework_benchmark.py``."""
    configure_runtime(
        dataset_root=dataset_root,
        coco_root=coco_root,
        helper_script=helper_script,
    )
    Path(config["run_dir"]).mkdir(parents=True, exist_ok=True)
    return train_task(config)
