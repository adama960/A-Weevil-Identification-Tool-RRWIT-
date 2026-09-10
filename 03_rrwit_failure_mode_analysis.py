#!/usr/bin/env python3
"""Summarize the final RRWIT failure modes from the frozen test audit.

This public-facing analysis uses the final manuscript failure taxonomy:

- Correct: species correct and localization IoU >= 0.50
- Misclassified: localized at IoU >= 0.50 but species incorrect
- Localization failure: a detection exists but selected IoU < 0.50
- No detection: no prediction survives the fixed confidence threshold

Optional EigenGradCAM results can be merged for diagnostic summaries of attribution
localization by prediction status and object size. The superseded multi-method
EigenCAM/Grad-CAM++/occlusion failure taxonomy is intentionally not used here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


LOGGER = logging.getLogger("rrwit.failure_modes")

SIZE_ORDER = ["Tiny (<1%)", "Small (1–<10%)", "Medium (10–<50%)", "Large (≥50%)"]
OUTCOME_ORDER = ["Correct", "Misclassified", "Localization failure", "No detection"]
EXPECTED_PUBLICATION_COUNTS = {
    "Correct": 1527,
    "Misclassified": 137,
    "Localization failure": 6,
    "No detection": 15,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize final RRWIT failure modes."
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        required=True,
        help="Complete single-object prediction audit produced by the EigenGradCAM workflow.",
    )
    parser.add_argument(
        "--eigengradcam-csv",
        type=Path,
        default=None,
        help="Optional EigenGradCAM image-level results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for failure-mode analysis outputs.",
    )
    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=0.75,
        help="Diagnostic threshold for confident wrong-species predictions. Default: 0.75",
    )
    parser.add_argument(
        "--skip-publication-qc",
        action="store_true",
        help="Skip frozen manuscript-specific row/count checks.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a streaming SHA-256 checksum."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    table_name: str,
) -> None:
    """Require a specific set of columns in an input table."""
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def validate_audit(audit: pd.DataFrame, publication_qc: bool) -> None:
    """Validate structural and optional manuscript-specific audit invariants."""
    require_columns(
        audit,
        {
            "file",
            "ground_truth_species",
            "predicted_species",
            "prediction_confidence",
            "predicted_gt_iou",
            "ground_truth_box_area_fraction",
            "size_stratum",
            "outcome",
            "prediction_correct",
        },
        "audit",
    )

    if audit["file"].isna().any() or audit["file"].duplicated().any():
        raise ValueError("Audit must contain exactly one nonmissing row per image path.")

    unexpected = sorted(set(audit["outcome"].dropna()) - set(OUTCOME_ORDER))
    if unexpected:
        raise ValueError(f"Unexpected audit outcomes: {unexpected}")

    if publication_qc:
        if len(audit) != 1685:
            raise RuntimeError(
                f"Publication QC expected 1,685 rows; observed {len(audit):,}."
            )
        observed = audit["outcome"].value_counts().to_dict()
        if observed != EXPECTED_PUBLICATION_COUNTS:
            raise RuntimeError(
                "Publication QC outcome counts differ from the frozen manuscript. "
                f"Expected {EXPECTED_PUBLICATION_COUNTS}; observed {observed}."
            )


def overall_failure_summary(audit: pd.DataFrame) -> pd.DataFrame:
    """Summarize the mutually exclusive primary failure categories."""
    total = len(audit)
    counts = audit["outcome"].value_counts().reindex(OUTCOME_ORDER, fill_value=0)
    summary = counts.rename("images").reset_index(names="outcome")
    summary["percent_of_audit"] = 100.0 * summary["images"] / total
    return summary


def size_failure_summary(audit: pd.DataFrame) -> pd.DataFrame:
    """Summarize outcomes and strict image-level accuracy by object-size stratum."""
    table = (
        audit.groupby(["size_stratum", "outcome"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=SIZE_ORDER, fill_value=0)
    )
    for outcome in OUTCOME_ORDER:
        if outcome not in table:
            table[outcome] = 0
    table = table[OUTCOME_ORDER].reset_index()
    table["audited_images"] = table[OUTCOME_ORDER].sum(axis=1)
    table["strict_image_level_accuracy"] = table["Correct"] / table["audited_images"]
    table["failure_rate"] = 1.0 - table["strict_image_level_accuracy"]
    return table


def species_failure_summary(audit: pd.DataFrame) -> pd.DataFrame:
    """Summarize RRWIT outcomes by ground-truth species."""
    counts = (
        audit.groupby(["ground_truth_species", "outcome"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    for outcome in OUTCOME_ORDER:
        if outcome not in counts:
            counts[outcome] = 0
    counts = counts[OUTCOME_ORDER].reset_index()
    counts["audited_images"] = counts[OUTCOME_ORDER].sum(axis=1)
    counts["strict_image_level_accuracy"] = counts["Correct"] / counts["audited_images"]
    counts["failure_rate"] = 1.0 - counts["strict_image_level_accuracy"]
    return counts.sort_values(
        ["failure_rate", "audited_images", "ground_truth_species"],
        ascending=[False, False, True],
    )


def misclassification_pairs(audit: pd.DataFrame) -> pd.DataFrame:
    """Rank localized wrong-species confusion pairs."""
    pairs = (
        audit.loc[audit["outcome"].eq("Misclassified")]
        .groupby(["ground_truth_species", "predicted_species"], observed=True)
        .agg(
            images=("file", "size"),
            mean_confidence=("prediction_confidence", "mean"),
            median_confidence=("prediction_confidence", "median"),
            mean_iou=("predicted_gt_iou", "mean"),
        )
        .reset_index()
        .sort_values(
            ["images", "ground_truth_species", "predicted_species"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    pairs.insert(0, "rank", np.arange(1, len(pairs) + 1))
    return pairs


def high_confidence_wrong_species(
    audit: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Identify confident wrong-species predictions as a diagnostic subset."""
    detected_wrong_species = (
        audit["predicted_species"].ne("NO_DETECTION")
        & audit["predicted_species"].ne(audit["ground_truth_species"])
        & audit["prediction_confidence"].ge(threshold)
    )
    columns = [
        "file",
        "ground_truth_species",
        "predicted_species",
        "prediction_confidence",
        "predicted_gt_iou",
        "size_stratum",
        "outcome",
    ]
    return (
        audit.loc[detected_wrong_species, columns]
        .sort_values(
            ["prediction_confidence", "ground_truth_species", "file"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )


def eigengradcam_diagnostics(
    audit: pd.DataFrame,
    xai: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize final EigenGradCAM localization by size and prediction status."""
    require_columns(
        xai,
        {
            "file",
            "prediction_status",
            "size_stratum",
            "gt_box_area_fraction",
            "gt_inside_box_enrichment",
        },
        "EigenGradCAM results",
    )
    if xai["file"].isna().any() or xai["file"].duplicated().any():
        raise ValueError("EigenGradCAM results must contain at most one row per image path.")

    missing_from_audit = sorted(set(xai["file"]) - set(audit["file"]))
    if missing_from_audit:
        raise RuntimeError(
            "EigenGradCAM contains image paths not present in the audit; "
            f"examples: {missing_from_audit[:5]}"
        )

    group_summary = (
        xai.groupby(["size_stratum", "prediction_status"], observed=False)[
            "gt_inside_box_enrichment"
        ]
        .agg(images="count", median_enrichment="median", mean_enrichment="mean")
        .reset_index()
    )

    correlations: list[dict[str, object]] = []
    for status in ("Correct", "Incorrect"):
        clean = xai.loc[
            xai["prediction_status"].eq(status),
            ["gt_box_area_fraction", "gt_inside_box_enrichment"],
        ].replace([np.inf, -np.inf], np.nan).dropna()
        rho, p_value = (
            spearmanr(clean["gt_box_area_fraction"], clean["gt_inside_box_enrichment"])
            if len(clean) >= 3
            else (np.nan, np.nan)
        )
        correlations.append(
            {
                "prediction_status": status,
                "images": len(clean),
                "spearman_rho": rho,
                "spearman_p_value_exploratory": p_value,
            }
        )
    return group_summary, pd.DataFrame(correlations)


def main() -> None:
    """Execute the final RRWIT failure-mode analysis."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not args.audit_csv.is_file():
        raise FileNotFoundError(f"Audit CSV does not exist: {args.audit_csv}")
    if args.eigengradcam_csv is not None and not args.eigengradcam_csv.is_file():
        raise FileNotFoundError(f"EigenGradCAM CSV does not exist: {args.eigengradcam_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(args.audit_csv, low_memory=False)
    validate_audit(audit, publication_qc=not args.skip_publication_qc)
    audit["size_stratum"] = pd.Categorical(audit["size_stratum"], SIZE_ORDER, ordered=True)

    overall = overall_failure_summary(audit)
    by_size = size_failure_summary(audit)
    by_species = species_failure_summary(audit)
    pairs = misclassification_pairs(audit)
    confident_wrong = high_confidence_wrong_species(
        audit,
        args.high_confidence_threshold,
    )

    overall.to_csv(args.output_dir / "RRWIT_failure_modes_overall.csv", index=False)
    by_size.to_csv(args.output_dir / "RRWIT_failure_modes_by_size.csv", index=False)
    by_species.to_csv(args.output_dir / "RRWIT_failure_modes_by_species.csv", index=False)
    pairs.to_csv(args.output_dir / "RRWIT_misclassification_pairs_ranked.csv", index=False)
    confident_wrong.to_csv(
        args.output_dir / "RRWIT_high_confidence_wrong_species.csv",
        index=False,
    )

    xai_summary_path = None
    xai_correlation_path = None
    if args.eigengradcam_csv is not None:
        xai = pd.read_csv(args.eigengradcam_csv, low_memory=False)
        xai_summary, xai_correlations = eigengradcam_diagnostics(audit, xai)
        xai_summary_path = args.output_dir / "RRWIT_EigenGradCAM_failure_mode_summary.csv"
        xai_correlation_path = args.output_dir / "RRWIT_EigenGradCAM_box_area_correlations.csv"
        xai_summary.to_csv(xai_summary_path, index=False)
        xai_correlations.to_csv(xai_correlation_path, index=False)

    qc = {
        "status": "PASS",
        "audit_csv": str(args.audit_csv),
        "audit_sha256": sha256_file(args.audit_csv),
        "audit_images": int(len(audit)),
        "unique_files": int(audit["file"].nunique()),
        "outcome_counts": {
            str(key): int(value) for key, value in audit["outcome"].value_counts().items()
        },
        "high_confidence_wrong_species_threshold": args.high_confidence_threshold,
        "high_confidence_wrong_species_images": int(len(confident_wrong)),
        "eigengradcam_csv": str(args.eigengradcam_csv) if args.eigengradcam_csv is not None else None,
        "eigengradcam_summary": str(xai_summary_path) if xai_summary_path is not None else None,
        "eigengradcam_correlations": str(xai_correlation_path) if xai_correlation_path is not None else None,
    }
    if args.eigengradcam_csv is not None:
        qc["eigengradcam_sha256"] = sha256_file(args.eigengradcam_csv)

    (args.output_dir / "RRWIT_failure_mode_QC.json").write_text(
        json.dumps(qc, indent=2) + "\n",
        encoding="utf-8",
    )

    LOGGER.info("Failure-mode analysis completed successfully.")
    LOGGER.info("Primary outcome counts: %s", qc["outcome_counts"])
    LOGGER.info(
        "High-confidence wrong-species detections (>= %.2f): %d",
        args.high_confidence_threshold,
        len(confident_wrong),
    )


if __name__ == "__main__":
    main()
