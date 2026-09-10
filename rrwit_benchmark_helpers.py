#!/usr/bin/env python3
"""Shared scientific helpers for the final eight-framework RRWIT benchmark.

This module consolidates the public portions of the original dataset-loader,
architecture-construction, pretrained-weight-transfer, and optimizer helpers used
by the RRWIT benchmark. It intentionally excludes internal preflight reporting,
release-freeze logic, Slurm metadata, machine-specific paths, and the RT-DETR
development branch that was not retained in the final manuscript benchmark.

The final public benchmark contains exactly eight detector configurations:
YOLOv5nu, YOLOv8n, RRWIT/YOLO11n, YOLO12n, YOLO26n, SSDLite320
MobileNetV3-Large, EfficientDet-D0, and Faster R-CNN
MobileNetV3-Large-320 FPN.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from effdet import create_model
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    ssdlite320_mobilenet_v3_large,
)

NUM_CLASSES = 14

# The training adapter sets these immediately before optimizer construction.
ACTIVE_MODEL = ""
ACTIVE_RECIPE: dict[str, Any] = {}


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to float32 RGB CHW in the closed interval [0, 1]."""
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255.0)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert absolute COCO xywh boxes to absolute xyxy coordinates."""
    output = boxes.clone().to(torch.float64)
    output[:, 2] = boxes[:, 0] + boxes[:, 2]
    output[:, 3] = boxes[:, 1] + boxes[:, 3]
    return output


def xyxy_to_cxcywh_normalized(
    boxes: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """Convert absolute xyxy boxes to normalized YOLO cxcywh coordinates."""
    result = torch.empty_like(boxes, dtype=torch.float64)
    result[:, 0] = (boxes[:, 0] + boxes[:, 2]) / (2.0 * width)
    result[:, 1] = (boxes[:, 1] + boxes[:, 3]) / (2.0 * height)
    result[:, 2] = (boxes[:, 2] - boxes[:, 0]) / width
    result[:, 3] = (boxes[:, 3] - boxes[:, 1]) / height
    return result


def validate_canonical(record: dict[str, Any]) -> None:
    """Validate the framework-neutral RRWIT image/target representation."""
    image = record["image"]
    target = record["target"]
    boxes = target["boxes"]
    labels = target["labels"]
    width, height = int(record["width"]), int(record["height"])

    if image.dtype != torch.float32 or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Image must be float32 RGB in CHW layout.")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Image tensor must be finite and within [0, 1].")
    if boxes.dtype != torch.float32 or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Canonical boxes must be float32 [N, 4] xyxy.")
    if labels.dtype != torch.int64 or labels.ndim != 1 or len(labels) != len(boxes):
        raise ValueError("Labels must be int64 [N] and align with boxes.")
    if len(boxes) == 0:
        raise ValueError("RRWIT benchmark records must contain at least one annotation.")
    if not torch.isfinite(boxes).all():
        raise ValueError("Bounding boxes contain non-finite coordinates.")
    if not ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])).all():
        raise ValueError("Bounding boxes must have positive width and height.")
    if not (
        (boxes[:, 0] >= 0)
        & (boxes[:, 1] >= 0)
        & (boxes[:, 2] <= width)
        & (boxes[:, 3] <= height)
    ).all():
        raise ValueError("Bounding boxes extend outside image boundaries.")
    if not ((labels >= 1) & (labels <= NUM_CLASSES)).all():
        raise ValueError(f"Foreground labels must be in 1..{NUM_CLASSES}.")
    if target["area"].dtype != torch.float32 or not (target["area"] > 0).all():
        raise ValueError("Annotation areas must be positive float32 values.")
    if target["iscrowd"].dtype != torch.int64 or not (target["iscrowd"] == 0).all():
        raise ValueError("RRWIT benchmark annotations require iscrowd=0.")


class FrozenCocoDataset(Dataset):
    """Load one frozen COCO partition into the shared RRWIT record contract."""

    def __init__(
        self,
        dataset_root: Path,
        annotation_json: Path,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        self.annotation_json = annotation_json.resolve()
        self.coco = COCO(str(self.annotation_json))
        self.ids = sorted(self.coco.getImgIds())
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id = self.ids[index]
        metadata = self.coco.loadImgs([image_id])[0]
        image_path = (self.dataset_root / metadata["file_name"]).resolve()

        if self.dataset_root not in image_path.parents or not image_path.is_file():
            raise FileNotFoundError(
                f"Image does not resolve safely from dataset root: {image_path}"
            )

        with Image.open(image_path) as opened:
            image = image_to_tensor(opened)

        height, width = image.shape[-2:]
        if (width, height) != (int(metadata["width"]), int(metadata["height"])):
            raise ValueError(
                f"Image dimensions disagree with COCO metadata: {image_path}"
            )

        annotations = self.coco.loadAnns(
            self.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        )
        annotations.sort(key=lambda item: int(item["id"]))

        xywh = torch.tensor(
            [ann["bbox"] for ann in annotations],
            dtype=torch.float32,
        ).reshape(-1, 4)
        boxes = xywh_to_xyxy(xywh).to(torch.float32)
        labels = torch.tensor(
            [ann["category_id"] for ann in annotations],
            dtype=torch.int64,
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": torch.tensor(
                [ann["area"] for ann in annotations],
                dtype=torch.float32,
            ),
            "iscrowd": torch.tensor(
                [ann.get("iscrowd", 0) for ann in annotations],
                dtype=torch.int64,
            ),
            "annotation_ids": torch.tensor(
                [ann["id"] for ann in annotations],
                dtype=torch.int64,
            ),
        }
        record = {
            "image": image,
            "target": target,
            "image_path": str(image_path),
            "file_name": metadata["file_name"],
            "width": width,
            "height": height,
        }

        validate_canonical(record)
        return self.transform(record) if self.transform else record


def clone_record(record: dict[str, Any]) -> dict[str, Any]:
    """Clone tensor content so transforms do not mutate the source record in place."""
    return {
        **record,
        "image": record["image"].clone(),
        "target": {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in record["target"].items()
        },
    }


def eval_identity(record: dict[str, Any]) -> dict[str, Any]:
    """Validation policy: deterministic conversion only, without augmentation."""
    return clone_record(record)


def collate_records(
    batch: list[dict[str, Any]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]], list[dict[str, Any]]]:
    """Collate variable-size detection targets without altering canonical records."""
    images = [item["image"] for item in batch]
    targets = [item["target"] for item in batch]
    return images, targets, batch


def make_adapters(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build framework-specific targets from one canonical COCO record.

    COCO is the annotation authority. TorchVision uses absolute xyxy with
    1-based foreground labels, YOLO uses normalized cxcywh with 0-based labels,
    and EfficientDet uses absolute yxyx with 1-based foreground labels.
    """
    boxes = record["target"]["boxes"].to(torch.float64)
    labels = record["target"]["labels"]
    width, height = int(record["width"]), int(record["height"])

    normalized = xyxy_to_cxcywh_normalized(boxes, width, height)
    xywh = boxes.clone()
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]

    return {
        "coco": {
            "boxes": xywh,
            "labels": labels.clone(),
            "box_format": "xywh_abs",
            "label_base": 1,
        },
        "yolo": {
            "boxes": normalized,
            "labels": labels - 1,
            "box_format": "cxcywh_norm",
            "label_base": 0,
        },
        "torchvision": {
            "boxes": boxes,
            "labels": labels.clone(),
            "box_format": "xyxy_abs",
            "label_base": 1,
        },
        "efficientdet": {
            "boxes": boxes[:, [1, 0, 3, 2]],
            "labels": labels.clone(),
            "box_format": "yxyx_abs",
            "label_base": 1,
        },
    }


def compatible_transfer(
    module: torch.nn.Module,
    source: dict[str, torch.Tensor],
) -> dict[str, int]:
    """Load shape-compatible pretrained tensors while resetting incompatible heads."""
    destination = module.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in destination and destination[key].shape == value.shape
    }
    mismatched = sum(
        key in destination and destination[key].shape != value.shape
        for key, value in source.items()
    )
    unexpected = sum(key not in destination for key in source)
    result = module.load_state_dict(compatible, strict=False)

    if not compatible:
        raise RuntimeError("No pretrained tensors were transferable.")

    return {
        "source_tensors": len(source),
        "transferred_tensors": len(compatible),
        "shape_mismatches": int(mismatched),
        "unexpected_source_tensors": int(unexpected),
        "missing_destination_tensors": len(result.missing_keys),
    }


def _normalization_types() -> tuple[type, ...]:
    """Return torch normalization module types for optimizer grouping."""
    return tuple(
        value
        for key, value in torch.nn.__dict__.items()
        if key.startswith("Norm") and isinstance(value, type)
    )


def _yolo_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Reproduce the public YOLO SGD grouping used in the benchmark audit."""
    normalization_types = _normalization_types()
    decay: list[torch.nn.Parameter] = []
    normalization: list[torch.nn.Parameter] = []
    bias: list[torch.nn.Parameter] = []
    seen: set[int] = set()

    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if name == "bias":
                bias.append(parameter)
            elif (
                isinstance(module, normalization_types)
                or "norm" in module.__class__.__name__.lower()
            ):
                normalization.append(parameter)
            else:
                decay.append(parameter)

    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if seen != expected:
        raise RuntimeError(f"{ACTIVE_MODEL}: incomplete YOLO optimizer grouping.")

    groups = [
        {"params": bias, "weight_decay": 0.0, "group_name": "bias"},
        {"params": decay, "weight_decay": weight_decay, "group_name": "decay_weight"},
        {"params": normalization, "weight_decay": 0.0, "group_name": "normalization"},
    ]
    groups = [group for group in groups if group["params"]]
    counts = {
        group["group_name"]: sum(parameter.numel() for parameter in group["params"])
        for group in groups
    }
    return groups, counts


def build_optimizer(
    model: torch.nn.Module,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    """Construct the frozen family-specific optimizer from ``ACTIVE_RECIPE``.

    The training adapter sets ``ACTIVE_MODEL`` and ``ACTIVE_RECIPE`` immediately
    before invoking this function. The helper supports only the eight retained
    public benchmark models.
    """
    if not ACTIVE_MODEL:
        raise RuntimeError("ACTIVE_MODEL must be set before optimizer construction.")
    if not ACTIVE_RECIPE:
        raise RuntimeError("ACTIVE_RECIPE must be set before optimizer construction.")

    spec = ACTIVE_RECIPE["optimizer"]

    if ACTIVE_MODEL.startswith("yolo"):
        groups, counts = _yolo_parameter_groups(
            model,
            float(spec["weight_decay"]),
        )
        optimizer = torch.optim.SGD(
            groups,
            lr=float(spec["lr0"]),
            momentum=float(spec["momentum"]),
            nesterov=bool(spec["nesterov"]),
        )

    elif spec["name"] == "SGD":
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            parameters,
            lr=float(spec["lr"]),
            momentum=float(spec["momentum"]),
            weight_decay=float(spec["weight_decay"]),
            nesterov=bool(spec["nesterov"]),
        )
        counts = {"all_trainable": sum(p.numel() for p in parameters)}

    elif spec["name"] == "AdamW":
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(spec["lr"]),
            betas=tuple(spec["betas"]),
            eps=float(spec["eps"]),
            weight_decay=float(spec["weight_decay"]),
        )
        counts = {"all_trainable": sum(p.numel() for p in parameters)}

    else:
        raise ValueError(
            f"Unsupported optimizer '{spec['name']}' for retained model {ACTIVE_MODEL}."
        )

    information = {
        "class": optimizer.__class__.__name__,
        "parameter_counts": counts,
        "initial_lrs": [float(group["lr"]) for group in optimizer.param_groups],
        "weight_decays": [
            float(group.get("weight_decay", 0.0))
            for group in optimizer.param_groups
        ],
    }
    return optimizer, information


def validate_recipe(model_name: str, recipe: dict[str, Any]) -> None:
    """Perform lightweight structural validation of a public training recipe."""
    if model_name not in {
        "yolov5nu",
        "yolov8n",
        "yolo11n_rrwit",
        "yolo12n",
        "yolo26n",
        "ssdlite320_mobilenet_v3_large",
        "efficientdet_d0",
        "fasterrcnn_mobilenet_v3_large_320_fpn",
    }:
        raise ValueError(f"Unsupported public benchmark model: {model_name}")

    required = {"optimizer", "scheduler", "warmup"}
    missing = required - set(recipe)
    if missing:
        raise ValueError(f"Training recipe is missing sections: {sorted(missing)}")
