#!/usr/bin/env python3
"""Run the final RRWIT EigenGradCAM analysis used for publication.

The script performs two connected analyses:

1. A complete single-object test audit that separates:
   - correct detections,
   - species misclassifications,
   - localization failures, and
   - no detections.

2. A deterministic, error-enriched EigenGradCAM analysis:
   - all detected errors are retained;
   - correctly detected images are limited to the three highest-confidence
     images per species-by-size stratum;
   - EigenGradCAM is calculated from YOLO11n layer 16;
   - attribution localization is quantified relative to ground-truth and
     predicted bounding boxes.

This public script intentionally excludes the earlier exploratory EigenCAM,
Grad-CAM++, occlusion, and plotting workflows.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import platform
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import ultralytics
from PIL import Image
from pytorch_grad_cam import EigenGradCAM
from scipy.stats import spearmanr
from ultralytics import YOLO


LOGGER = logging.getLogger("rrwit.eigengradcam")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SIZE_ORDER = ["Tiny (<1%)", "Small (1–<10%)", "Medium (10–<50%)", "Large (≥50%)"]
STATUS_ORDER = ["Correct", "Incorrect"]

DEFAULT_SEED = 20260808
DEFAULT_IMAGE_SIZE = 640
DEFAULT_CONFIDENCE = 0.25
DEFAULT_NMS_IOU = 0.70
DEFAULT_CORRECT_IOU = 0.50
DEFAULT_RAW_MATCH_IOU = 0.50
DEFAULT_TARGET_LAYER = 16
DEFAULT_MAX_CORRECT_PER_SPECIES_SIZE = 3


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the final RRWIT EigenGradCAM analysis."
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to RRWIT.v1.0.0.pt.")
    parser.add_argument("--test-images", type=Path, required=True, help="Directory containing test images.")
    parser.add_argument("--test-labels", type=Path, required=True, help="Directory containing YOLO-format test labels.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for analysis outputs.")
    parser.add_argument("--device", default=None, help="Inference device, e.g. '0' or 'cpu'.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU)
    parser.add_argument("--correct-iou", type=float, default=DEFAULT_CORRECT_IOU)
    parser.add_argument("--raw-match-iou", type=float, default=DEFAULT_RAW_MATCH_IOU)
    parser.add_argument("--target-layer", type=int, default=DEFAULT_TARGET_LAYER)
    parser.add_argument(
        "--max-correct-per-species-size",
        type=int,
        default=DEFAULT_MAX_CORRECT_PER_SPECIES_SIZE,
    )
    parser.add_argument(
        "--skip-publication-qc",
        action="store_true",
        help="Skip manuscript-specific row/count checks while retaining structural QC.",
    )
    return parser.parse_args()


def resolve_device(requested: str | None) -> str | int:
    """Return an Ultralytics-compatible device selector."""
    if requested is not None:
        return requested
    return 0 if torch.cuda.is_available() else "cpu"


def torch_device_from_selector(selector: str | int) -> torch.device:
    """Return a torch.device corresponding to the Ultralytics selector."""
    if str(selector).lower() == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{selector}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_zero_one(array: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Min-max normalize finite values, returning zeros for a constant map."""
    values = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(values)
    output = np.zeros_like(values)
    if not finite.any():
        return output
    low = float(values[finite].min())
    high = float(values[finite].max())
    if high - low > epsilon:
        output[finite] = (values[finite] - low) / (high - low)
    return output


def xywhn_to_xyxy(row: list[str], width: int, height: int) -> np.ndarray:
    """Convert one normalized YOLO label row to clipped pixel xyxy coordinates."""
    cx, cy = float(row[1]) * width, float(row[2]) * height
    box_w, box_h = float(row[3]) * width, float(row[4]) * height
    box = np.array(
        [cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2],
        dtype=np.float32,
    )
    box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"Invalid box after clipping: {box.tolist()}")
    return box


def box_iou_numpy(
    boxes: np.ndarray,
    reference: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """Calculate IoU between N xyxy boxes and one reference box."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    ref = np.asarray(reference, dtype=np.float32)
    ix1 = np.maximum(boxes[:, 0], ref[0])
    iy1 = np.maximum(boxes[:, 1], ref[1])
    ix2 = np.minimum(boxes[:, 2], ref[2])
    iy2 = np.minimum(boxes[:, 3], ref[3])
    intersection = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    box_area = (
        np.maximum(0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    )
    ref_area = max(0, ref[2] - ref[0]) * max(0, ref[3] - ref[1])
    return intersection / (box_area + ref_area - intersection + epsilon)


def map_original_box_to_input(
    box: np.ndarray,
    original_h: int,
    original_w: int,
    input_h: int,
    input_w: int,
) -> np.ndarray:
    """Map an original-image box into Ultralytics letterboxed input space."""
    gain = min(input_w / original_w, input_h / original_h)
    resized_w, resized_h = round(original_w * gain), round(original_h * gain)
    pad_x, pad_y = (input_w - resized_w) / 2, (input_h - resized_h) / 2
    mapped = np.asarray(box, dtype=np.float32).copy()
    mapped[[0, 2]] = mapped[[0, 2]] * gain + pad_x
    mapped[[1, 3]] = mapped[[1, 3]] * gain + pad_y
    return mapped


def restore_cam(
    cam: np.ndarray,
    original_h: int,
    original_w: int,
    input_h: int,
    input_w: int,
) -> np.ndarray:
    """Remove letterbox padding and resize a CAM to original-image coordinates."""
    gain = min(input_w / original_w, input_h / original_h)
    resized_w, resized_h = round(original_w * gain), round(original_h * gain)
    left = max(0, int(round((input_w - resized_w) / 2 - 0.1)))
    top = max(0, int(round((input_h - resized_h) / 2 - 0.1)))
    cropped = np.asarray(cam)[
        top:min(input_h, top + resized_h),
        left:min(input_w, left + resized_w),
    ]
    if cropped.size == 0:
        raise RuntimeError("Inverse letterbox produced an empty CAM.")
    resized = cv2.resize(cropped, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    return normalize_zero_one(resized)


def localization_metrics(
    attribution: np.ndarray,
    box: np.ndarray,
    prefix: str,
) -> dict[str, float | bool]:
    """Measure attribution concentration relative to one bounding box."""
    values = np.clip(np.nan_to_num(np.asarray(attribution, dtype=np.float32)), 0, None)
    height, width = values.shape
    x1, y1, x2, y2 = np.rint(box).astype(int)
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid {prefix} box.")

    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    total = float(values.sum())
    inside = float(values[mask].sum())
    box_area_fraction = float(mask.mean())
    attribution_fraction_inside = inside / total if total > 0 else np.nan
    peak = np.unravel_index(np.argmax(values), values.shape) if total > 0 else None

    return {
        f"{prefix}_box_area_fraction": box_area_fraction,
        f"{prefix}_attribution_fraction_inside": attribution_fraction_inside,
        f"{prefix}_inside_box_enrichment": (
            attribution_fraction_inside / box_area_fraction
            if box_area_fraction > 0
            else np.nan
        ),
        f"{prefix}_mean_inside": float(values[mask].mean()),
        f"{prefix}_mean_outside": float(values[~mask].mean()) if (~mask).any() else np.nan,
        f"pointing_game_{prefix}_hit": bool(mask[peak]) if peak is not None else np.nan,
    }


def size_stratum(area_fraction: float) -> str:
    """Assign the manuscript object-size category from ground-truth box area."""
    if area_fraction < 0.01:
        return "Tiny (<1%)"
    if area_fraction < 0.10:
        return "Small (1–<10%)"
    if area_fraction < 0.50:
        return "Medium (10–<50%)"
    return "Large (≥50%)"


def load_single_object_paths(
    test_images: Path,
    test_labels: Path,
) -> tuple[list[Path], list[dict[str, object]]]:
    """Identify test images with exactly one valid YOLO annotation."""
    all_images = sorted(
        path for path in test_images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not all_images:
        raise RuntimeError(f"No supported test images found in {test_images}")

    included: list[Path] = []
    excluded: list[dict[str, object]] = []
    for image_path in all_images:
        label_path = test_labels / f"{image_path.stem}.txt"
        if not label_path.is_file():
            excluded.append({"file": str(image_path), "reason": "missing_label"})
            continue
        rows = [
            line.split()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != 1:
            excluded.append(
                {
                    "file": str(image_path),
                    "reason": "not_single_object",
                    "annotation_count": len(rows),
                }
            )
            continue
        if len(rows[0]) != 5:
            excluded.append({"file": str(image_path), "reason": "malformed_label"})
            continue
        included.append(image_path)
    return included, excluded


def audit_test_images(
    model: YOLO,
    class_names: list[str],
    image_paths: list[Path],
    test_labels: Path,
    image_size: int,
    confidence: float,
    nms_iou: float,
    correct_iou: float,
    device: str | int,
) -> pd.DataFrame:
    """Build the complete single-object RRWIT prediction audit."""
    records: list[dict[str, object]] = []
    for number, image_path in enumerate(image_paths, start=1):
        label_path = test_labels / f"{image_path.stem}.txt"
        label = [
            line.split()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][0]
        gt_class_id = int(label[0])
        if gt_class_id < 0 or gt_class_id >= len(class_names):
            raise ValueError(f"Invalid class id {gt_class_id} in {label_path}")

        with Image.open(image_path) as image:
            width, height = image.size
        gt_box = xywhn_to_xyxy(label, width, height)

        result = model.predict(
            source=str(image_path),
            imgsz=image_size,
            conf=confidence,
            iou=nms_iou,
            device=device,
            verbose=False,
        )[0]
        detections = 0 if result.boxes is None else len(result.boxes)
        record: dict[str, object] = {
            "file": str(image_path),
            "ground_truth_class_id": gt_class_id,
            "ground_truth_species": str(class_names[gt_class_id]),
            "image_width": width,
            "image_height": height,
            "ground_truth_x1": float(gt_box[0]),
            "ground_truth_y1": float(gt_box[1]),
            "ground_truth_x2": float(gt_box[2]),
            "ground_truth_y2": float(gt_box[3]),
            "ground_truth_box_area_fraction": float(label[3]) * float(label[4]),
            "number_of_detections": detections,
        }

        if detections == 0:
            record.update(
                {
                    "selected_detection_index": np.nan,
                    "predicted_class_id": np.nan,
                    "predicted_species": "NO_DETECTION",
                    "prediction_confidence": 0.0,
                    "predicted_x1": np.nan,
                    "predicted_y1": np.nan,
                    "predicted_x2": np.nan,
                    "predicted_y2": np.nan,
                    "predicted_gt_iou": np.nan,
                    "class_correct": False,
                    "localization_correct": False,
                    "prediction_correct": False,
                    "outcome": "No detection",
                }
            )
        else:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            ious = box_iou_numpy(boxes, gt_box)
            selected = int(np.argmax(ious))
            pred_box = boxes[selected].astype(np.float32)
            pred_id = int(result.boxes.cls[selected].item())
            class_correct = pred_id == gt_class_id
            localization_correct = float(ious[selected]) >= correct_iou

            if class_correct and localization_correct:
                outcome = "Correct"
            elif not localization_correct:
                outcome = "Localization failure"
            else:
                outcome = "Misclassified"

            record.update(
                {
                    "selected_detection_index": selected,
                    "predicted_class_id": pred_id,
                    "predicted_species": str(class_names[pred_id]),
                    "prediction_confidence": float(result.boxes.conf[selected].item()),
                    "predicted_x1": float(pred_box[0]),
                    "predicted_y1": float(pred_box[1]),
                    "predicted_x2": float(pred_box[2]),
                    "predicted_y2": float(pred_box[3]),
                    "predicted_gt_iou": float(ious[selected]),
                    "class_correct": bool(class_correct),
                    "localization_correct": bool(localization_correct),
                    "prediction_correct": bool(class_correct and localization_correct),
                    "outcome": outcome,
                }
            )

        records.append(record)
        if number % 64 == 0 or number == len(image_paths):
            LOGGER.info("Audited %d/%d images", number, len(image_paths))

    audit = pd.DataFrame(records)
    audit["size_stratum"] = audit["ground_truth_box_area_fraction"].map(size_stratum)
    audit["size_stratum"] = pd.Categorical(audit["size_stratum"], SIZE_ORDER, ordered=True)
    audit["prediction_status"] = np.where(audit["prediction_correct"], "Correct", "Incorrect")
    if audit["file"].duplicated().any():
        raise RuntimeError("Complete audit contains duplicate image paths.")
    return audit


def select_xai_subset(
    audit: pd.DataFrame,
    max_correct_per_species_size: int,
) -> pd.DataFrame:
    """Retain all detected errors and a deterministic subset of correct images."""
    eligible = audit[audit["outcome"] != "No detection"].copy()
    selected_parts: list[pd.DataFrame] = []
    for (_, _, status), group in eligible.groupby(
        ["ground_truth_species", "size_stratum", "prediction_status"],
        observed=True,
        sort=True,
    ):
        ordered = group.sort_values(["prediction_confidence", "file"], ascending=[False, True])
        if status == "Correct":
            ordered = ordered.head(max_correct_per_species_size)
        selected_parts.append(ordered)

    selection = (
        pd.concat(selected_parts, ignore_index=True)
        .sort_values(["ground_truth_species", "size_stratum", "prediction_status", "file"])
        .reset_index(drop=True)
    )
    selection["selection_reason"] = np.where(
        selection["prediction_status"].eq("Incorrect"),
        "all_detected_errors_retained",
        f"up_to_{max_correct_per_species_size}_highest_confidence_correct_per_species_size",
    )
    if selection["file"].duplicated().any():
        raise RuntimeError("An image was selected for EigenGradCAM more than once.")
    return selection


class DecodedWrapper(nn.Module):
    """Expose the decoded three-dimensional YOLO prediction tensor to CAM."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output = self.model(tensor)
        if torch.is_tensor(output):
            decoded = output
        elif isinstance(output, (tuple, list)):
            decoded = next((item for item in output if torch.is_tensor(item)), None)
        else:
            decoded = None
        if decoded is None or decoded.ndim != 3:
            raise TypeError(f"Could not obtain a 3D decoded tensor; output={type(output)}")
        return decoded


def xywh_to_xyxy_torch(boxes: torch.Tensor) -> torch.Tensor:
    """Convert center-format torch boxes to xyxy."""
    cx, cy, width, height = boxes.unbind(dim=1)
    return torch.stack(
        [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
        dim=1,
    )


def torch_iou_against_reference(
    boxes: torch.Tensor,
    reference: torch.Tensor,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Calculate IoU between candidate boxes and one reference box."""
    ref = reference.reshape(1, 4)
    inter_w = (
        torch.minimum(boxes[:, 2], ref[:, 2]) - torch.maximum(boxes[:, 0], ref[:, 0])
    ).clamp(min=0)
    inter_h = (
        torch.minimum(boxes[:, 3], ref[:, 3]) - torch.maximum(boxes[:, 1], ref[:, 1])
    ).clamp(min=0)
    intersection = inter_w * inter_h
    box_area = (
        (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    )
    ref_area = (
        (ref[:, 2] - ref[:, 0]).clamp(min=0)
        * (ref[:, 3] - ref[:, 1]).clamp(min=0)
    )
    return intersection / (box_area + ref_area - intersection + epsilon)


class StrictDetectionTarget:
    """Target the raw candidate spatially matching one audited post-NMS box."""

    def __init__(
        self,
        class_id: int,
        reference_box_input: np.ndarray,
        number_of_classes: int,
        minimum_iou: float,
    ):
        self.class_id = int(class_id)
        self.reference_box = torch.as_tensor(reference_box_input, dtype=torch.float32)
        self.number_of_classes = int(number_of_classes)
        self.minimum_iou = float(minimum_iou)
        self.last_matched_iou = np.nan

    def __call__(self, per_image_output: torch.Tensor) -> torch.Tensor:
        if per_image_output.ndim != 2:
            raise ValueError(f"Expected 2D per-image output; got {tuple(per_image_output.shape)}")

        expected_channels = 4 + self.number_of_classes
        if per_image_output.shape[0] == expected_channels:
            decoded = per_image_output
        elif per_image_output.shape[1] == expected_channels:
            decoded = per_image_output.T
        else:
            raise ValueError(
                f"Neither decoded axis equals 4 + nc = {expected_channels}: "
                f"{tuple(per_image_output.shape)}"
            )

        boxes = xywh_to_xyxy_torch(decoded[:4].T)
        scores = decoded[4 : 4 + self.number_of_classes].T
        reference = self.reference_box.to(boxes.device, boxes.dtype)
        ious = torch_iou_against_reference(boxes, reference)
        index = int(torch.argmax(ious.detach()).item())
        self.last_matched_iou = float(ious[index].detach().cpu())
        if self.last_matched_iou < self.minimum_iou:
            raise RuntimeError(
                f"Raw-candidate match IoU {self.last_matched_iou:.4f} "
                f"< {self.minimum_iou:.2f}; no fallback used."
            )
        return scores[index, self.class_id]


def native_yolo_preprocess(
    model: YOLO,
    image_path: Path,
    image_size: int,
    confidence: float,
    nms_iou: float,
    device_selector: str | int,
    torch_device: torch.device,
) -> tuple[object, np.ndarray, torch.Tensor]:
    """Return post-NMS results, RGB pixels, and a gradient-safe YOLO tensor."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"OpenCV could not read {image_path}")

    result = model.predict(
        source=str(image_path),
        imgsz=image_size,
        conf=confidence,
        iou=nms_iou,
        device=device_selector,
        verbose=False,
    )[0]

    predictor_tensor = model.predictor.preprocess([bgr.copy()])
    with torch.inference_mode(False):
        tensor = predictor_tensor.detach().clone().to(torch_device, torch.float32).contiguous()
    if torch.is_inference(tensor):
        raise RuntimeError("CAM input remains an inference tensor.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return result, rgb, tensor


def compute_eigengradcam(
    cam_model: nn.Module,
    target_layer: nn.Module,
    tensor: torch.Tensor,
    target: StrictDetectionTarget,
) -> np.ndarray:
    """Compute one normalized EigenGradCAM map."""
    with EigenGradCAM(model=cam_model, target_layers=[target_layer]) as cam:
        batch = cam(
            input_tensor=tensor,
            targets=[target],
            aug_smooth=False,
            eigen_smooth=False,
        )
    if batch is None or len(batch) != 1:
        raise RuntimeError("EigenGradCAM did not return exactly one map.")
    return normalize_zero_one(batch[0])


def run_eigengradcam(
    inference_model: YOLO,
    selection: pd.DataFrame,
    class_names: list[str],
    model_path: Path,
    image_size: int,
    confidence: float,
    nms_iou: float,
    raw_match_iou: float,
    target_layer_index: int,
    device_selector: str | int,
    torch_device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run EigenGradCAM for the deterministic XAI subset."""
    cam_yolo = YOLO(str(model_path))
    torch_model = cam_yolo.model.to(torch_device).float().eval()
    for parameter in torch_model.parameters():
        parameter.requires_grad_(True)

    model_layers = torch_model.model
    if target_layer_index >= len(model_layers):
        raise IndexError(
            f"Target layer {target_layer_index} is unavailable; model has {len(model_layers)} layers."
        )
    target_layer = model_layers[target_layer_index]

    detect_head = model_layers[-1]
    if hasattr(detect_head, "dynamic"):
        detect_head.dynamic = True
    if hasattr(detect_head, "shape"):
        detect_head.shape = None

    cam_model = DecodedWrapper(torch_model).to(torch_device).eval()
    with torch.inference_mode(False):
        diagnostic_output = cam_model(
            torch.zeros(
                (1, 3, image_size, image_size),
                device=torch_device,
                dtype=torch.float32,
            )
        )
    LOGGER.info(
        "CAM output shape: %s; target layer %d (%s)",
        tuple(diagnostic_output.shape),
        target_layer_index,
        type(target_layer).__name__,
    )

    xai_records: list[dict[str, object]] = []
    failure_records: list[dict[str, object]] = []
    for row_number, metadata in selection.iterrows():
        image_path = Path(metadata["file"])
        try:
            result, rgb, tensor = native_yolo_preprocess(
                inference_model,
                image_path,
                image_size,
                confidence,
                nms_iou,
                device_selector,
                torch_device,
            )
            if result.boxes is None or len(result.boxes) == 0:
                raise RuntimeError("No detection remained during the independent XAI run.")

            gt_box = np.array(
                [
                    metadata["ground_truth_x1"],
                    metadata["ground_truth_y1"],
                    metadata["ground_truth_x2"],
                    metadata["ground_truth_y2"],
                ],
                dtype=np.float32,
            )
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            selected = int(np.argmax(box_iou_numpy(boxes, gt_box)))
            pred_box = boxes[selected].astype(np.float32)
            pred_id = int(result.boxes.cls[selected].item())
            prediction_confidence = float(result.boxes.conf[selected].item())

            height, width = rgb.shape[:2]
            input_h, input_w = tensor.shape[-2:]
            pred_box_input = map_original_box_to_input(
                pred_box, height, width, input_h, input_w
            )
            target = StrictDetectionTarget(
                class_id=pred_id,
                reference_box_input=pred_box_input,
                number_of_classes=len(class_names),
                minimum_iou=raw_match_iou,
            )
            cam_input = compute_eigengradcam(cam_model, target_layer, tensor, target)
            cam_original = restore_cam(cam_input, height, width, input_h, input_w)

            xai_records.append(
                {
                    "file": str(image_path),
                    "ground_truth_species": metadata["ground_truth_species"],
                    "predicted_species": str(class_names[pred_id]),
                    "outcome": metadata["outcome"],
                    "prediction_status": metadata["prediction_status"],
                    "prediction_confidence": prediction_confidence,
                    "size_stratum": str(metadata["size_stratum"]),
                    "target_layer_index": target_layer_index,
                    "target_raw_candidate_iou": target.last_matched_iou,
                    "predicted_gt_iou": float(box_iou_numpy([pred_box], gt_box)[0]),
                    **localization_metrics(cam_original, gt_box, "gt"),
                    **localization_metrics(cam_original, pred_box, "pred"),
                }
            )
        except Exception as error:  # Preserve per-image analysis failures for auditability.
            failure_records.append(
                {
                    "file": str(image_path),
                    "ground_truth_species": metadata["ground_truth_species"],
                    "size_stratum": str(metadata["size_stratum"]),
                    "prediction_status": metadata["prediction_status"],
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(limit=5),
                }
            )
        finally:
            if torch_device.type == "cuda" and (row_number + 1) % 8 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        if (row_number + 1) % 16 == 0 or row_number + 1 == len(selection):
            LOGGER.info(
                "EigenGradCAM processed %d/%d selected images",
                row_number + 1,
                len(selection),
            )

    xai_df = pd.DataFrame(xai_records)
    failures_df = pd.DataFrame(failure_records)
    successful = set(xai_df["file"]) if not xai_df.empty else set()
    failed = set(failures_df["file"]) if not failures_df.empty else set()
    selected_files = set(selection["file"])
    if successful & failed:
        raise RuntimeError("An image appears in both success and failure tables.")
    if selected_files != successful | failed:
        raise RuntimeError("Selected-image accounting failed.")
    if not xai_df.empty and xai_df["file"].duplicated().any():
        raise RuntimeError("EigenGradCAM results contain duplicate images.")
    return xai_df, failures_df


def build_xai_summaries(
    xai_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize enrichment by size/status and compute object-size correlations."""
    group_summary = (
        xai_df.groupby(["size_stratum", "prediction_status"], observed=False)[
            "gt_inside_box_enrichment"
        ]
        .agg(images="count", median_enrichment="median")
        .reset_index()
    )

    correlation_records: list[dict[str, object]] = []
    for status in STATUS_ORDER:
        clean = xai_df.loc[
            xai_df["prediction_status"].eq(status),
            ["gt_box_area_fraction", "gt_inside_box_enrichment"],
        ].dropna()
        rho, p_value = (
            spearmanr(clean.iloc[:, 0], clean.iloc[:, 1])
            if len(clean) >= 3
            else (np.nan, np.nan)
        )
        correlation_records.append(
            {
                "prediction_status": status,
                "images": len(clean),
                "spearman_rho": rho,
                "spearman_p_value_exploratory": p_value,
            }
        )
    return group_summary, pd.DataFrame(correlation_records)


def publication_qc(audit: pd.DataFrame, xai_df: pd.DataFrame) -> None:
    """Check the frozen manuscript-level audit counts and final XAI subset size."""
    expected_counts = {
        "Correct": 1527,
        "Misclassified": 137,
        "Localization failure": 6,
        "No detection": 15,
    }
    if len(audit) != 1685:
        raise RuntimeError(
            f"Publication QC expected 1,685 single-object images; observed {len(audit):,}."
        )
    observed_counts = audit["outcome"].value_counts().to_dict()
    if observed_counts != expected_counts:
        raise RuntimeError(
            "Publication QC outcome counts differ from the frozen manuscript. "
            f"Expected {expected_counts}; observed {observed_counts}."
        )
    if len(xai_df) != 258:
        raise RuntimeError(
            f"Publication QC expected 258 successful EigenGradCAM images; observed {len(xai_df):,}."
        )


def main() -> None:
    """Execute the complete public EigenGradCAM analysis."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    for required in (args.model, args.test_images, args.test_labels):
        if not required.exists():
            raise FileNotFoundError(f"Required input does not exist: {required}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device_selector = resolve_device(args.device)
    torch_device = torch_device_from_selector(device_selector)

    rrwit = YOLO(str(args.model))
    class_names = rrwit.names
    if isinstance(class_names, dict):
        class_names = [class_names[index] for index in range(len(class_names))]
    if len(class_names) != 14:
        raise ValueError(f"Expected 14 RRWIT classes; observed {len(class_names)}.")

    manifest = {
        "analysis": "Final RRWIT EigenGradCAM analysis",
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "test_images": str(args.test_images),
        "test_labels": str(args.test_labels),
        "seed": args.seed,
        "device": str(device_selector),
        "image_size": args.image_size,
        "confidence_threshold": args.confidence,
        "nms_iou": args.nms_iou,
        "correct_localization_iou": args.correct_iou,
        "raw_candidate_minimum_iou": args.raw_match_iou,
        "target_layer_index": args.target_layer,
        "selection_rule": (
            "all detected errors plus up to "
            f"{args.max_correct_per_species_size} highest-confidence correct images "
            "per species-by-size stratum"
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    single_object_paths, excluded = load_single_object_paths(args.test_images, args.test_labels)
    pd.DataFrame(excluded).to_csv(
        args.output_dir / "test_images_excluded_from_single_object_analysis.csv",
        index=False,
    )

    audit = audit_test_images(
        model=rrwit,
        class_names=class_names,
        image_paths=single_object_paths,
        test_labels=args.test_labels,
        image_size=args.image_size,
        confidence=args.confidence,
        nms_iou=args.nms_iou,
        correct_iou=args.correct_iou,
        device=device_selector,
    )
    audit.to_csv(
        args.output_dir / "complete_single_object_prediction_audit.csv",
        index=False,
    )

    size_performance = (
        audit.groupby(["size_stratum", "outcome"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(SIZE_ORDER, fill_value=0)
        .reset_index()
    )
    for column in ("Correct", "Misclassified", "Localization failure", "No detection"):
        if column not in size_performance:
            size_performance[column] = 0
    size_performance["audited_images"] = size_performance[
        ["Correct", "Misclassified", "Localization failure", "No detection"]
    ].sum(axis=1)
    size_performance["strict_image_level_accuracy"] = (
        size_performance["Correct"] / size_performance["audited_images"]
    )
    size_performance.to_csv(
        args.output_dir / "complete_audit_performance_by_size.csv",
        index=False,
    )

    selection = select_xai_subset(audit, args.max_correct_per_species_size)
    selection.to_csv(args.output_dir / "eigengradcam_xai_selection.csv", index=False)

    xai_df, failures_df = run_eigengradcam(
        inference_model=rrwit,
        selection=selection,
        class_names=class_names,
        model_path=args.model,
        image_size=args.image_size,
        confidence=args.confidence,
        nms_iou=args.nms_iou,
        raw_match_iou=args.raw_match_iou,
        target_layer_index=args.target_layer,
        device_selector=device_selector,
        torch_device=torch_device,
    )
    xai_df.to_csv(args.output_dir / "eigengradcam_image_level_results.csv", index=False)
    failures_df.to_csv(args.output_dir / "eigengradcam_failures.csv", index=False)

    group_summary, correlations = build_xai_summaries(xai_df)
    group_summary.to_csv(args.output_dir / "eigengradcam_group_summary.csv", index=False)
    correlations.to_csv(args.output_dir / "eigengradcam_box_area_correlations.csv", index=False)

    if not args.skip_publication_qc:
        publication_qc(audit, xai_df)

    final_qc = {
        "status": "PASS",
        "single_object_audit_images": int(audit["file"].nunique()),
        "audit_outcomes": {
            str(key): int(value) for key, value in audit["outcome"].value_counts().items()
        },
        "selected_xai_images": int(selection["file"].nunique()),
        "successful_xai_images": int(xai_df["file"].nunique()),
        "failed_xai_images": int(len(failures_df)),
        "target_raw_candidate_iou_all_at_least_threshold": bool(
            (xai_df["target_raw_candidate_iou"] >= args.raw_match_iou).all()
        ),
    }
    (args.output_dir / "FINAL_INTEGRITY_QC.json").write_text(
        json.dumps(final_qc, indent=2) + "\n",
        encoding="utf-8",
    )

    LOGGER.info("Final EigenGradCAM analysis completed successfully.")
    LOGGER.info("Audit outcomes: %s", final_qc["audit_outcomes"])
    LOGGER.info(
        "EigenGradCAM: %d successful, %d failed",
        final_qc["successful_xai_images"],
        final_qc["failed_xai_images"],
    )


if __name__ == "__main__":
    main()
