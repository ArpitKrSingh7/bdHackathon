#!/usr/bin/env python3
"""
scripts/data_quality_audit.py
-----------------------------
Extended reproducible data-quality audit: schema/types, missingness, binary
domain verification, finite/positive geometry, exact duplicates, repeated
signatures with conflicting labels, and Tukey IQR outlier flags.

Saves machine-readable results to artifacts/data_quality_audit.json.
Does NOT automatically remove any rows or columns.
"""

import sys
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew


def run_data_quality_audit(
    data_path="data/raw/ad.data",
    names_path="data/raw/ad.names",
    output_path="artifacts/data_quality_audit.json",
):
    d_path = Path(data_path)
    n_path = Path(names_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not d_path.exists():
        print(f"Error: {d_path.resolve()} not found", file=sys.stderr)
        sys.exit(1)

    results = {}

    # ── Environment ──
    import scipy
    results["environment"] = {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    print(f"Python {results['environment']['python']}, "
          f"pandas {results['environment']['pandas']}, "
          f"scipy {results['environment']['scipy']}")

    # ── Parse feature names from ad.names ──
    feature_names = []
    declared_types = []
    if n_path.exists():
        with open(n_path, encoding="latin-1") as f:
            for line in f:
                match = re.match(
                    r"\s*([^|][^:]*):\s*(continuous|0,1)\.",
                    line, flags=re.IGNORECASE
                )
                if match:
                    feature_names.append(match.group(1).strip())
                    declared_types.append(match.group(2).strip().lower())
    results["schema"] = {
        "names_file": str(n_path),
        "parsed_feature_count": len(feature_names),
        "declared_continuous": sum(1 for t in declared_types if t == "continuous"),
        "declared_binary": sum(1 for t in declared_types if t == "0,1"),
    }
    print(f"Parsed {len(feature_names)} features from {n_path.name}: "
          f"{results['schema']['declared_continuous']} continuous, "
          f"{results['schema']['declared_binary']} binary (0,1)")

    # ── Load raw data ──
    df = pd.read_csv(d_path, header=None, skipinitialspace=True,
                      na_values=["?"], low_memory=False)
    n_rows, n_cols = df.shape
    target = df.iloc[:, -1].astype(str).str.strip().str.rstrip(".")
    features = df.iloc[:, :-1]

    results["dimensions"] = {"rows": int(n_rows), "columns": int(n_cols),
                             "input_features": int(n_cols - 1)}
    print(f"\nLoaded {n_rows:,} rows × {n_cols:,} cols")

    # ── Target class distribution ──
    class_counts = target.value_counts().to_dict()
    results["target"] = {
        "column_index": n_cols - 1,
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "imbalance_ratio": round(class_counts["nonad"] / class_counts["ad"], 4),
    }
    print(f"Classes: {class_counts}, ratio {results['target']['imbalance_ratio']}")

    # ── 1. Schema & type verification ──
    # Geometry columns (0,1,2) should be numeric; binary columns (3..1557) should be 0/1
    geom_cols = [0, 1, 2]
    binary_col_start = 3
    binary_col_end = n_cols - 2  # last input column index

    # Check geometry columns are numeric after NA removal
    geom_type_ok = {}
    for col in geom_cols:
        series = features[col].dropna()
        numeric = pd.to_numeric(series, errors="coerce")
        non_numeric_count = int(numeric.isna().sum())
        geom_type_ok[feature_names[col] if col < len(feature_names) else f"col_{col}"] = {
            "non_missing": int(len(series)),
            "non_numeric_values": non_numeric_count,
            "all_numeric": non_numeric_count == 0,
        }
    results["geometry_type_check"] = geom_type_ok
    print(f"\nGeometry type check: all numeric = "
          f"{all(v['all_numeric'] for v in geom_type_ok.values())}")

    # Check binary columns: every non-missing value must be in {0, 1}
    binary_domain_violations = {}
    binary_cols_checked = 0
    for col in range(binary_col_start, binary_col_end + 1):
        series = features[col].dropna()
        unique_vals = set(series.unique())
        valid_domain = unique_vals.issubset({0, 1, 0.0, 1.0})
        binary_cols_checked += 1
        if not valid_domain:
            col_name = feature_names[col] if col < len(feature_names) else f"col_{col}"
            binary_domain_violations[col_name] = {
                "column_index": col,
                "unique_values": sorted([str(v) for v in unique_vals])[:10],
            }

    results["binary_domain_check"] = {
        "columns_checked": binary_cols_checked,
        "violations": len(binary_domain_violations),
        "all_restricted_to_0_1": len(binary_domain_violations) == 0,
        "violation_details": binary_domain_violations if binary_domain_violations else "none",
    }
    print(f"Binary domain check: {binary_cols_checked} columns checked, "
          f"{len(binary_domain_violations)} violations, "
          f"all restricted to {{0,1}} = {len(binary_domain_violations) == 0}")

    # ── 2. Missingness audit ──
    missing = features.isna().sum()
    cols_with_missing = missing[missing > 0]
    missing_detail = {}
    for col_idx, miss_count in cols_with_missing.items():
        col_name = feature_names[col_idx] if col_idx < len(feature_names) else f"col_{col_idx}"
        missing_detail[col_name] = {
            "column_index": int(col_idx),
            "missing_count": int(miss_count),
            "missing_pct": round(miss_count / n_rows * 100, 4),
        }

    geom_missing_rows = int(features.iloc[:, :3].isna().any(axis=1).sum())
    any_missing_rows = int(features.isna().any(axis=1).sum())

    results["missingness"] = {
        "columns_with_missing": len(cols_with_missing),
        "per_column": missing_detail,
        "rows_missing_geometry": geom_missing_rows,
        "rows_missing_any_input": any_missing_rows,
    }
    print(f"\nMissing: {len(cols_with_missing)} columns, "
          f"{geom_missing_rows} rows missing geometry, "
          f"{any_missing_rows} rows missing any input")

    # ── 3. Finite and positive geometry values ──
    geom_checks = {}
    for col, name in [(0, "height"), (1, "width"), (2, "aratio")]:
        series = pd.to_numeric(features[col], errors="coerce").dropna()
        finite_mask = np.isfinite(series)
        positive_mask = series > 0
        geom_checks[name] = {
            "non_missing": int(len(series)),
            "all_finite": bool(finite_mask.all()),
            "non_finite_count": int((~finite_mask).sum()),
            "all_positive": bool(positive_mask.all()),
            "non_positive_count": int((~positive_mask).sum()),
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "skewness_scipy": round(float(skew(series)), 4),
            "skewness_pandas": round(float(series.skew()), 4),
        }
    results["geometry_finite_positive"] = geom_checks
    all_finite_positive = all(
        v["all_finite"] and v["all_positive"] for v in geom_checks.values()
    )
    print(f"Geometry finite & positive: {all_finite_positive}")
    for name, checks in geom_checks.items():
        print(f"  {name}: n={checks['non_missing']}, "
              f"range=[{checks['min']}, {checks['max']}], "
              f"median={checks['median']}, skew={checks['skewness_scipy']}")

    # ── 4. Tukey IQR outlier flags ──
    # Flag rows that fall outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] for any geometry feature.
    # These are flags only. We document why they do or do not justify removal.
    iqr_flags = {}
    any_flagged_mask = pd.Series(False, index=df.index)
    for col, name in [(0, "height"), (1, "width"), (2, "aratio")]:
        series = pd.to_numeric(features[col], errors="coerce")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        flagged = ((series < lower_fence) | (series > upper_fence)) & series.notna()
        any_flagged_mask = any_flagged_mask | flagged
        iqr_flags[name] = {
            "Q1": round(float(q1), 4),
            "Q3": round(float(q3), 4),
            "IQR": round(float(iqr), 4),
            "lower_fence": round(float(lower_fence), 4),
            "upper_fence": round(float(upper_fence), 4),
            "flagged_count": int(flagged.sum()),
            "flagged_pct": round(float(flagged.sum() / n_rows * 100), 4),
        }
        print(f"  IQR flags for {name}: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, "
              f"fences=[{lower_fence:.2f}, {upper_fence:.2f}], "
              f"flagged={flagged.sum()}")

    total_flagged = int(any_flagged_mask.sum())
    results["iqr_outlier_flags"] = {
        "per_feature": iqr_flags,
        "rows_flagged_by_at_least_one_feature": total_flagged,
        "rows_flagged_pct": round(total_flagged / n_rows * 100, 4),
        "automatic_removal": False,
        "justification": (
            "Tukey IQR fences are review flags, not proof that a record is corrupt. "
            "The raw audit found finite positive values in all three geometry "
            "fields, but the dataset has no external validation label for whether "
            "an unusual size is an error. The strong right skew of aratio makes "
            "distribution-based fences indicators of unusualness rather than "
            "ground truth. These flags motivate robust transformations for "
            "distance-based methods. No rows are removed based on IQR flags alone."
        ),
    }
    print(f"\nTotal rows flagged by at least one geometry IQR fence: {total_flagged}")

    # ── 5. Duplicates and conflicting labels ──
    exact_dups = int(df.duplicated().sum())
    feat_str = features.astype(str).agg(",".join, axis=1)
    repeated_sig_all = int(feat_str.duplicated(keep=False).sum())
    feat_input_dups = int(features.duplicated().sum())

    df_check = pd.DataFrame({"sig": feat_str, "target": target})
    sig_label_counts = df_check.groupby("sig")["target"].nunique()
    conflicting_sigs = sig_label_counts[sig_label_counts > 1].index
    conflicting_rows = df_check[df_check["sig"].isin(conflicting_sigs)]
    conflict_groups = []
    for s in conflicting_sigs:
        sub = conflicting_rows[conflicting_rows["sig"] == s]
        conflict_groups.append({
            "rows": int(len(sub)),
            "class_counts": {k: int(v) for k, v in sub["target"].value_counts().items()},
        })

    results["duplicates"] = {
        "exact_duplicate_rows_including_class": exact_dups,
        "feature_only_duplicate_rows": feat_input_dups,
        "rows_in_repeated_signatures": repeated_sig_all,
        "conflicting_label_groups": len(conflict_groups),
        "conflicting_label_total_rows": int(len(conflicting_rows)),
        "conflict_details": conflict_groups,
    }
    print(f"\nExact duplicates (with class): {exact_dups}")
    print(f"Feature-only duplicates: {feat_input_dups}")
    print(f"Rows in repeated signatures: {repeated_sig_all}")
    print(f"Conflicting label groups: {len(conflict_groups)} "
          f"({len(conflicting_rows)} rows)")

    # ── 6. Feature variance ──
    nunique = features.nunique(dropna=True)
    constant_cols = int((nunique <= 1).sum())
    results["feature_variance"] = {
        "constant_columns": constant_cols,
        "all_features_vary": constant_cols == 0,
    }
    print(f"\nConstant columns: {constant_cols}")

    # ── Save ──
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    run_data_quality_audit()
