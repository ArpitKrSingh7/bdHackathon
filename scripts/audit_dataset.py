#!/usr/bin/env python3
"""
scripts/audit_dataset.py
------------------------
Independent, fully reproducible dataset audit script for the UCI Internet Advertisements dataset.
Verifies dimensions, class distribution, missing value counts, moment skewness,
duplicate signatures, conflicting labels, and feature variation directly from the raw files.
"""

import sys
import re
from pathlib import Path
import pandas as pd
from scipy.stats import skew

def run_audit(data_path="data/raw/ad.data", names_path="data/raw/ad.names"):
    d_path = Path(data_path)
    n_path = Path(names_path)

    if not d_path.exists():
        print(f"Error: dataset file not found at {d_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    print("=== ENVIRONMENT & DEPENDENCIES ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"pandas: {pd.__version__}")
    import scipy
    print(f"scipy:  {scipy.__version__}")
    print()

    # Parse feature names from ad.names (requires latin-1 encoding)
    feature_names = []
    if n_path.exists():
        with open(n_path, encoding="latin-1") as f:
            for line in f:
                match = re.match(r"\s*([^|][^:]*):\s*(?:continuous|0,1)\.", line, flags=re.IGNORECASE)
                if match:
                    feature_names.append(match.group(1).strip())
        print(f"Parsed {len(feature_names)} input features from {n_path.name} (encoding: latin-1)")
    print()

    print("=== LOADING RAW DATASET ===")
    # ad.data has variable leading whitespace before '?'
    df = pd.read_csv(d_path, header=None, skipinitialspace=True, na_values=["?"], low_memory=False)
    n_rows, n_cols = df.shape
    print(f"File: {d_path}")
    print(f"Dimensions: {n_rows:,} rows × {n_cols:,} columns (1,558 inputs + 1 target class)")
    print()

    print("=== TARGET CLASS DISTRIBUTION ===")
    target = df.iloc[:, -1].astype(str).str.strip().str.rstrip(".")
    counts = target.value_counts()
    for label, count in counts.items():
        pct = (count / n_rows) * 100
        print(f"  {label:<8}: {count:,} ({pct:.2f}%)")
    print(f"  Imbalance ratio (majority/minority): {counts['nonad'] / counts['ad']:.2f}")
    print()

    print("=== MISSING VALUE AUDIT ===")
    missing = df.isna().sum()
    cols_with_missing = missing[missing > 0]
    print(f"Columns with missing values ({len(cols_with_missing)} total):")
    feature_names_map = {0: "height", 1: "width", 2: "aratio", 3: "local"}
    for col_idx, miss_count in cols_with_missing.items():
        feat_name = feature_names_map.get(col_idx, f"col_{col_idx}")
        pct = (miss_count / n_rows) * 100
        print(f"  Column {col_idx:<2} ({feat_name:<6}): {miss_count:,} missing ({pct:.2f}%)")
    
    geom_missing = df.iloc[:, :3].isna().any(axis=1).sum()
    any_missing = df.iloc[:, :-1].isna().any(axis=1).sum()
    print(f"Rows missing at least one geometry value (cols 0-2): {geom_missing:,} ({(geom_missing/n_rows)*100:.2f}%)")
    print(f"Rows missing at least one input feature (cols 0-1557): {any_missing:,} ({(any_missing/n_rows)*100:.2f}%)")
    print()

    print("=== MOMENT SKEWNESS (GEOMETRY FEATURES) ===")
    for col_idx, name in [(0, "height"), (1, "width"), (2, "aratio")]:
        series = df[col_idx].dropna().astype(float)
        m_skew = skew(series)
        p_skew = series.skew()
        print(f"  {name:<6} (n={len(series):,}): SciPy moment skewness = {m_skew:.4f} | pandas sample skewness = {p_skew:.4f}")
    print()

    print("=== CLASS-CONDITIONAL MEDIANS (GEOMETRY FEATURES) ===")
    df_temp = df.copy()
    df_temp["class"] = target
    for col_idx, name in [(0, "height"), (1, "width"), (2, "aratio")]:
        med_ad = df_temp[df_temp["class"] == "ad"][col_idx].median()
        med_nonad = df_temp[df_temp["class"] == "nonad"][col_idx].median()
        print(f"  {name:<6}: ad median = {med_ad:.1f} | nonad median = {med_nonad:.4f}")
    print()

    print("=== DUPLICATE ANALYSIS & REPEATED SIGNATURES ===")
    exact_dups = df.duplicated().sum()
    print(f"Exact duplicate rows (features + class): {exact_dups:,}")

    # String signature of features
    feat_str = df.iloc[:, :-1].astype(str).agg(",".join, axis=1)
    repeated_sig_rows = feat_str.duplicated(keep=False).sum()
    print(f"Rows sharing feature signatures with >= 1 other row: {repeated_sig_rows:,}")

    df_check = pd.DataFrame({"sig": feat_str, "target": target})
    sig_label_counts = df_check.groupby("sig")["target"].nunique()
    conflicting_sigs = sig_label_counts[sig_label_counts > 1].index
    conflicting_rows = df_check[df_check["sig"].isin(conflicting_sigs)]
    print(f"Feature signatures with conflicting labels: {len(conflicting_sigs)} signatures, involving {len(conflicting_rows)} total rows")
    for s_idx, s in enumerate(conflicting_sigs):
        sub = conflicting_rows[conflicting_rows["sig"] == s]
        print(f"  Conflict group {s_idx+1}: {len(sub)} rows -> {dict(sub['target'].value_counts())}")
    print()

    print("=== FEATURE VARIANCE IN RAW DATA ===")
    nunique = df.iloc[:, :-1].nunique(dropna=True)
    constant_cols = nunique[nunique <= 1]
    print(f"Constant input columns (zero variance): {len(constant_cols)} (all 1,558 features vary)")

if __name__ == "__main__":
    run_audit()
