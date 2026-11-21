"""Save a non-destructive cleaned copy and task-specific preprocessing record."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hackathon_utils import RAW_DATA_DIR, feature_signatures, load_ads, raw_file_manifest

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "processed"


def save_preprocessing_artifacts() -> dict:
    """Normalize labels/types, preserve missingness/rows, and save configuration."""
    data, features, numeric, binary = load_ads(RAW_DATA_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_path = OUTPUT_DIR / "internet_ads_cleaned.csv"
    data.to_csv(cleaned_path, index=False, na_rep="")

    signatures = feature_signatures(data, features)
    counts = signatures.value_counts()
    target_counts = data.groupby(signatures, sort=False)["class"].nunique()
    signature_rows = pd.DataFrame({
        "row_position": range(len(data)),
        "feature_signature": signatures.to_numpy(dtype="uint64"),
        "class": data["class"].to_numpy(),
    })
    group_sizes = signature_rows.groupby("feature_signature")["row_position"].transform("size")
    signature_rows["signature_group_size"] = group_sizes
    signature_rows["conflicting_label_group"] = signature_rows["feature_signature"].map(target_counts.gt(1))
    signature_path = OUTPUT_DIR / "feature_signature_ledger.csv"
    signature_rows.to_csv(signature_path, index=False)

    config = {
        "source": {
            "dataset": "UCI Internet Advertisements",
            "raw_directory": "data/raw",
            "raw_sha256_manifest": raw_file_manifest(RAW_DATA_DIR),
        },
        "normalized_copy": {
            "path": str(cleaned_path.relative_to(PROJECT_DIR)),
            "rows": int(len(data)),
            "columns_including_target": int(data.shape[1]),
            "target_labels": {str(k): int(v) for k, v in data["class"].value_counts().items()},
            "label_whitespace_and_trailing_period_trimmed": True,
            "feature_types_from_ad_names": {
                "continuous": numeric,
                "binary": binary,
            },
            "missing_values_preserved": True,
            "rows_or_repeated_signatures_dropped": False,
        },
        "repeated_feature_signatures": {
            "signature_method": "pandas.util.hash_pandas_object over all input feature columns, index=False",
            "distinct_signatures": int(counts.size),
            "groups_with_repeated_rows": int((counts > 1).sum()),
            "rows_in_repeated_groups": int(counts[counts > 1].sum()),
            "extra_rows_beyond_first_per_signature": int((counts - 1).clip(lower=0).sum()),
            "groups_with_conflicting_labels": int(target_counts.gt(1).sum()),
            "ledger": str(signature_path.relative_to(PROJECT_DIR)),
        },
        "task_specific_transformations": {
            "association_mining": {
                "representation": "one transaction per raw row; observed binary 1 values become items",
                "numeric_geometry": "excluded; no ungrounded binning",
                "missing_local": "explicit local=unknown token",
                "minimum_item_support": 0.05,
                "repeated_rows": "retained as separate transactions",
            },
            "decision_tree": {
                "split": "group feature signatures before fitting; keep every row in one partition",
                "continuous_missing": "median fitted on training rows only",
                "binary_missing": "most-frequent value fitted on training rows only",
                "scaling": "none; tree split thresholds do not require scaling",
            },
            "dbscan": {
                "features": numeric,
                "missing": "median imputation on unlabeled geometry table for descriptive clustering",
                "transformation": "log1p then RobustScaler (median/IQR)",
                "distance": "Euclidean in transformed geometry space",
                "target_used_to_fit": False,
            },
        },
    }
    config_path = OUTPUT_DIR / "preprocessing_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "rows": int(len(data)),
        "features": int(len(features)),
        "missing_cells": int(data[features].isna().sum().sum()),
        "distinct_signatures": int(counts.size),
        "repeated_groups": int((counts > 1).sum()),
        "conflicting_label_groups": int(target_counts.gt(1).sum()),
        "cleaned_path": str(cleaned_path),
        "config_path": str(config_path),
        "signature_ledger_path": str(signature_path),
    }


if __name__ == "__main__":
    print(json.dumps(save_preprocessing_artifacts(), indent=2))
