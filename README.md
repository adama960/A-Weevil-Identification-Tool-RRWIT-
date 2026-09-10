# RRWIT analysis main code snippets and Env configuration files

This repository contains a curated subset of code supporting Automated detection and classification of morphologically similar weevil species using AI for precision pest monitoring article. Rostrum Recognition: A Weevil Identification Tool (RRWIT) is a lightweight object-detection framework for distinguishing 14 weevil taxa, including economically important boll and pecan weevils, using more than 16,000 expert-verified annotated images..

## Included workflow files

1. `01_train_original_rrwit_yolo11n.py` — historical COCO-pretrained YOLO11n RRWIT training.
2. `02_rrwit_eigengradcam_analysis.py` — final EigenGradCAM audit and quantitative attribution analysis.
3. `03_rrwit_failure_mode_analysis.py` — final RRWIT failure-mode summaries.
4. `04_compare_random_vs_pretrained.py` — matched random-initialization versus COCO-pretrained YOLO11n experiment across seeds 42, 123, and 2025.
5. `05_rrwit_architecture_ablation.py` — validation-only A0–A3 P2/CBAM architecture screen.
6. `06_train_eight_framework_benchmark.py` — public entry point for the final eight-framework, three-seed benchmark.
7. `rrwit_eight_framework_training_adapter.py` — family-specific training and validation-selection logic.
8. `rrwit_benchmark_helpers.py` — canonical COCO loader, framework target adapters, TorchVision/EfficientDet constructors, pretrained-weight transfer, and optimizer construction used by the benchmark.
9. `07_evaluate_locked_test_benchmark.py` — validation-threshold lock and held-out-test evaluation from standardized COCO predictions.
10. `08_compute_model_footprint.py` — parameter memory and theoretical forward-operation counts.
11. `09_export_fp32_onnx.py` — FP32 ONNX export for the four deployment candidates.
12. `10_validate_fp32_onnx_parity.py` — validation-only canonical PyTorch versus ONNX parity checks.
13. `11_quantize_int8_models.py` — common static INT8 QDQ/MinMax post-training quantization using a frozen 512-image training cohort.
14. `12_validate_int8_accuracy.py` — validation-only FP32 versus INT8 accuracy-retention qualification.
15. `13_efficientdet_int8_diagnostics.py` — EfficientDet-D0 signedness/per-channel diagnostic matrix.
16. `14_benchmark_cpu_runtime.py` — controlled CPU ONNX Runtime benchmark.

The final framework benchmark contains exactly **eight** detector configurations: YOLOv5nu, YOLOv8n, RRWIT (YOLO11n), YOLO12n, YOLO26n, SSDLite320 MobileNetV3-Large, EfficientDet-D0, and Faster R-CNN MobileNetV3-Large-320 FPN.

## Reproducibility boundaries

Validation is used for checkpoint selection, architecture screening, and operating-threshold selection. The held-out test split is not used for model fitting or reselection. The benchmark uses seeds 42, 123, and 2025. YOLO-family models use 640 × 640 inputs, EfficientDet-D0 uses 512 × 512, and the SSDLite/Faster R-CNN configurations use 320 × 320. Quantization calibration and CPU timing use training-only cohorts. The public XAI workflow uses EigenGradCAM only.

The benchmark helper module preserves a common COCO annotation authority and framework-specific target conventions: YOLO uses normalized 0-based labels, TorchVision uses absolute xyxy boxes with 1-based foreground labels, and EfficientDet uses absolute yxyx targets with 1-based foreground labels. Final predictions are normalized to a common COCO evaluation boundary before comparison.

## Environment

The public repository uses a compact dependency specification rather than the original machine-specific Conda export. Create the environment from the repository root with:

```bash
conda env create -f environment.yml
conda activate rrwit
```

Alternatively, install the direct Python dependencies into an existing Python 3.10 environment:

```bash
python -m pip install -r requirements.txt
```

Versions explicitly recorded in the archived RRWIT environment are pinned for NumPy, OpenCV, pandas, Pillow, PyYAML, SciPy, PyTorch, TorchVision, Ultralytics, and `ultralytics-thop`. The final public scripts also require `effdet`, `grad-cam` (imported as `pytorch_grad_cam`), `onnx`, `onnxruntime`, and `pycocotools`; those packages were not present in the supplied archived environment export, so this repository does not invent exact historical version numbers for them.

The original explicit Conda package dump is intentionally not distributed because it contains platform-specific build records and is less portable than the curated environment above. CUDA toolkit/runtime components are likewise not manually pinned in `requirements.txt`; PyTorch installation should be matched to the user's supported CUDA/CPU platform.

## Paths and external artifacts

No institution-specific filesystem paths, usernames, job identifiers, host names, access tokens, or private URLs are embedded. Dataset locations, pretrained checkpoints, trained checkpoints, and output directories are supplied by the user through command-line arguments or task configuration files.

Large datasets and model artifacts are distributed separately through the study data/model repositories.
