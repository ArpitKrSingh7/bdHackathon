"""Shared, validated loader for the UCI Internet Advertisements assignment."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

BDHACKATHON_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BDHACKATHON_DIR / "data" / "raw"
UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/51/internet%2Badvertisements"
UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/51/internet+advertisements.zip"
TARGET_COLUMN = "class"


def parse_attribute_definitions(names_path: Path) -> Tuple[List[str], Dict[str, str]]:
    """Read declared feature names/types from UCI's ad.names file."""
    feature_names: List[str] = []
    feature_types: Dict[str, str] = {}
    declaration = re.compile(r"^\s*([^|][^:]*):\s*(continuous|0,1)\.", re.IGNORECASE)
    for line in names_path.read_text(encoding="latin-1").splitlines():
        match = declaration.match(line)
        if match:
            name = match.group(1).strip()
            kind = match.group(2).lower()
            feature_names.append(name)
            feature_types[name] = "continuous" if kind == "continuous" else "binary"
    if not feature_names:
        raise ValueError(f"No feature declarations parsed from {names_path}")
    return feature_names, feature_types


def load_ads(raw_dir: Path | str = RAW_DATA_DIR):
    """Load raw UCI files without changing them.

    Returns `(data, feature_names, numeric_features, binary_features)`. The final
    target token's whitespace and trailing period are normalized. Unknown values
    remain missing; no imputation or feature selection occurs here.
    """
    raw_dir = Path(raw_dir)
    data_path = raw_dir / "ad.data"
    names_path = raw_dir / "ad.names"
    missing_files = [str(path) for path in (data_path, names_path) if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Missing raw UCI files: " + ", ".join(missing_files) +
            f". Download the official archive: {UCI_ARCHIVE_URL}"
        )

    feature_names, feature_types = parse_attribute_definitions(names_path)
    data = pd.read_csv(
        data_path,
        header=None,
        names=feature_names + [TARGET_COLUMN],
        skipinitialspace=True,
        na_values=["?"],
    )
    if data.shape[1] != len(feature_names) + 1:
        raise ValueError(
            f"Expected {len(feature_names) + 1} columns, received {data.shape[1]}"
        )

    data[TARGET_COLUMN] = (
        data[TARGET_COLUMN].astype("string").str.strip().str.rstrip(".")
    )
    labels = set(data[TARGET_COLUMN].dropna().unique())
    if labels != {"ad", "nonad"}:
        raise ValueError(f"Unexpected target labels in raw file: {sorted(labels)}")

    numeric_features = [name for name in feature_names if feature_types[name] == "continuous"]
    binary_features = [name for name in feature_names if feature_types[name] == "binary"]
    for feature in feature_names:
        converted = pd.to_numeric(data[feature], errors="coerce")
        invalid = data[feature].notna() & converted.isna()
        if invalid.any():
            examples = data.loc[invalid, feature].head(5).tolist()
            raise ValueError(f"Non-numeric values in {feature}: {examples}")
        data[feature] = converted

    invalid_binary = {
        feature: sorted(set(data[feature].dropna().unique()) - {0, 1})
        for feature in binary_features
        if not set(data[feature].dropna().unique()).issubset({0, 1})
    }
    if invalid_binary:
        raise ValueError(f"Binary features contain values outside 0/1: {invalid_binary}")

    return data, feature_names, numeric_features, binary_features


def feature_signatures(data: pd.DataFrame, feature_names: List[str]) -> pd.Series:
    """Stable-within-environment signatures for duplicate-aware grouped splits."""
    return pd.util.hash_pandas_object(data[feature_names], index=False)


def raw_file_manifest(raw_dir: Path | str = RAW_DATA_DIR) -> List[dict]:
    """Return size and SHA-256 for the source files without altering them."""
    raw_dir = Path(raw_dir)
    manifest = []
    for name in ("ad.data", "ad.names", "ad.DOCUMENTATION"):
        path = raw_dir / name
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append({"file": name, "bytes": path.stat().st_size, "sha256": digest})
    return manifest
