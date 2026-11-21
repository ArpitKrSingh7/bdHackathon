#!/usr/bin/env python3
"""Synthetic row-replication scaling benchmark for local CART vs Spark CART.

The experiment repeats the real, already-imputed training rows from the
group-aware baseline split. This measures runtime scaling under an explicitly
synthetic workload. It adds no independent observations and cannot support
claims about real-world accuracy, memory break-even, or cluster scale-out.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.tree import DecisionTreeClassifier

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from classification_clustering import TREE_PARAMETERS  # noqa: E402

BASE_ARRAYS = PROJECT_DIR / "artifacts" / "models" / "tree_shared_split.npz"
DEFAULT_SIZES = (3279, 10000, 25000, 50000, 100000)
REPEATS = 2


def _expanded_training_data(base_x: sparse.csr_matrix, base_y: np.ndarray, n_rows: int):
    repeats, remainder = divmod(n_rows, len(base_y))
    pieces = [base_x] * repeats
    labels = [base_y] * repeats
    if remainder:
        pieces.append(base_x[:remainder])
        labels.append(base_y[:remainder])
    start = time.perf_counter()
    X = sparse.vstack(pieces, format="csr", dtype=np.float64)
    y = np.concatenate(labels).astype(np.int32, copy=False)
    expansion_seconds = time.perf_counter() - start
    return X, y, expansion_seconds


def _local_fit(X, y, repeats: int) -> list[float]:
    timings = []
    for _ in range(repeats):
        model = DecisionTreeClassifier(**TREE_PARAMETERS)
        start = time.perf_counter()
        model.fit(X, y)
        timings.append(time.perf_counter() - start)
    return timings


def _spark_fit(spark, X, y, repeats: int) -> dict:
    from pyspark.ml.classification import DecisionTreeClassifier as SparkTree
    from pyspark.ml.linalg import VectorUDT, Vectors
    from pyspark.sql import Row
    from pyspark.sql.types import DoubleType, StructField, StructType

    n_rows, n_features = X.shape
    class_counts = np.bincount(y, minlength=2)
    total = int(class_counts.sum())
    weights = {label: total / (2.0 * int(count)) for label, count in enumerate(class_counts)}
    schema = StructType([
        StructField("features", VectorUDT(), nullable=False),
        StructField("label", DoubleType(), nullable=False),
        StructField("class_weight", DoubleType(), nullable=False),
    ])

    start = time.perf_counter()
    records = []
    for row_idx in range(n_rows):
        start_idx, end_idx = X.indptr[row_idx], X.indptr[row_idx + 1]
        indices = X.indices[start_idx:end_idx].tolist()
        values = X.data[start_idx:end_idx].tolist()
        records.append(Row(
            features=Vectors.sparse(n_features, indices, values),
            label=float(y[row_idx]),
            class_weight=float(weights[int(y[row_idx])]),
        ))
    vector_construction_seconds = time.perf_counter() - start

    start = time.perf_counter()
    frame = spark.createDataFrame(records, schema=schema).cache()
    materialized_rows = frame.count()
    dataframe_materialization_seconds = time.perf_counter() - start
    if materialized_rows != n_rows:
        raise RuntimeError(f"Expected {n_rows} Spark records, received {materialized_rows}")

    params = {
        "impurity": TREE_PARAMETERS["criterion"],
        "maxDepth": int(TREE_PARAMETERS["max_depth"]),
        "minInstancesPerNode": int(TREE_PARAMETERS["min_samples_leaf"]),
        "maxBins": 32,
        "seed": int(TREE_PARAMETERS["random_state"]),
        "weightCol": "class_weight",
    }
    timings = []
    for _ in range(repeats):
        model = SparkTree(**params)
        start = time.perf_counter()
        model.fit(frame)
        timings.append(time.perf_counter() - start)
    frame.unpersist(blocking=True)
    return {
        "vector_construction_seconds": vector_construction_seconds,
        "dataframe_materialization_seconds": dataframe_materialization_seconds,
        "fit_seconds_repeats": timings,
        "fit_seconds_median": statistics.median(timings),
        "materialized_rows": int(materialized_rows),
        "parameters": params,
    }


def run(sizes: tuple[int, ...], repeats: int = REPEATS) -> dict:
    if not BASE_ARRAYS.is_file():
        raise FileNotFoundError(f"Expected the existing prepared grouped split: {BASE_ARRAYS}")
    with np.load(BASE_ARRAYS, allow_pickle=False) as saved:
        base_x = np.asarray(saved["X_train"], dtype=np.float64)
        base_y = np.asarray(saved["y_train"], dtype=np.int32)
    base_x = sparse.csr_matrix(base_x)
    base_x.eliminate_zeros()
    if set(np.unique(base_y)) != {0, 1}:
        raise ValueError("Prepared training labels must contain both 0 and 1")
    sizes = tuple(sorted(set(int(size) for size in sizes)))
    if any(size < len(base_y) for size in sizes):
        raise ValueError(f"Each requested size must be at least {len(base_y)} base training rows")

    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    from pyspark.sql import SparkSession
    startup_start = time.perf_counter()
    spark_error = None
    try:
        spark = (
            SparkSession.builder.master("local[2]")
            .appName("InternetAds-Synthetic-Scale-Benchmark")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
        startup_seconds = time.perf_counter() - startup_start
        spark.sparkContext.setLogLevel("ERROR")
    except Exception as exc:
        spark = None
        startup_seconds = None
        spark_error = f"{type(exc).__name__}: {exc}"

    measurements = []
    try:
        for size in sizes:
            X, y, expansion_seconds = _expanded_training_data(base_x, base_y, size)
            local_times = _local_fit(X, y, repeats)
            spark_result = _spark_fit(spark, X, y, repeats) if spark is not None else {
                "vector_construction_seconds": None,
                "dataframe_materialization_seconds": None,
                "fit_seconds_repeats": [],
                "fit_seconds_median": None,
                "materialized_rows": None,
                "parameters": None,
            }
            local_fit = statistics.median(local_times)
            local_total = expansion_seconds + local_fit
            spark_total = None if spark is None else (
                expansion_seconds
                + spark_result["vector_construction_seconds"]
                + spark_result["dataframe_materialization_seconds"]
                + spark_result["fit_seconds_median"]
            )
            measurements.append({
                "synthetic_training_rows": int(size),
                "base_training_rows": int(len(base_y)),
                "feature_count": int(X.shape[1]),
                "nonzero_fraction": float(X.nnz / (X.shape[0] * X.shape[1])),
                "local_csr_expansion_seconds": expansion_seconds,
                "local_fit_seconds_repeats": local_times,
                "local_fit_seconds_median": local_fit,
                "local_expand_plus_fit_seconds": local_total,
                **spark_result,
                "spark_expand_convert_materialize_fit_seconds_excluding_startup": spark_total,
                "fit_only_speedup_local_over_spark": (
                    None if spark is None else local_fit / spark_result["fit_seconds_median"]
                ),
                "end_to_end_speedup_local_over_spark": (
                    None if spark is None else local_total / spark_total
                ),
                "predictive_metrics_measured": False,
            })
            print(json.dumps({
                "rows": size,
                "local_fit_s": round(local_fit, 4),
                "spark_fit_s": None if spark is None else round(spark_result["fit_seconds_median"], 4),
                "spark_total_s_excl_startup": None if spark_total is None else round(spark_total, 4),
                "fit_speedup_local_over_spark": measurements[-1]["fit_only_speedup_local_over_spark"],
                "end_to_end_speedup_local_over_spark": measurements[-1]["end_to_end_speedup_local_over_spark"],
            }), flush=True)
            del X, y
    finally:
        if spark is not None:
            spark.stop()

    payload = {
        "benchmark": "Local scikit-learn CART vs PySpark MLlib DecisionTreeClassifier",
        "status": "completed" if spark is not None else "local_curve_completed_spark_blocked_by_sandbox",
        "synthetic_scale_test": True,
        "construction": "Repeated rows from the existing group-aware training partition; repetitions add no independent observations.",
        "base_training_rows": int(len(base_y)),
        "features": int(base_x.shape[1]),
        "spark_master": "local[2]",
        "spark_startup_seconds_separate": startup_seconds,
        "spark_startup_error": spark_error,
        "repeats_per_size": repeats,
        "local_parameters": TREE_PARAMETERS,
        "spark_parameters": measurements[0]["parameters"] if measurements else None,
        "interpretation_limits": [
            "No accuracy or generalization metrics are estimated by this benchmark.",
            "The replicated rows are a synthetic throughput workload, not new Internet Ads observations.",
            "local[2] is one machine, not a multi-node cluster.",
            "Any observed crossover applies only to this benchmark design and must not be generalized to real new data.",
        ],
        "measurements": measurements,
    }
    output = PROJECT_DIR / "artifacts" / "spark"
    output.mkdir(parents=True, exist_ok=True)
    (output / "scaled_tree_benchmark.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(measurements).drop(columns=["local_fit_seconds_repeats", "fit_seconds_repeats"]).to_csv(
        output / "scaled_tree_benchmark.csv", index=False
    )
    frame = pd.DataFrame(measurements)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(frame["synthetic_training_rows"], frame["local_fit_seconds_median"], marker="o", label="Local CART fit")
    if frame["fit_seconds_median"].notna().any():
        ax.plot(frame["synthetic_training_rows"], frame["fit_seconds_median"], marker="o", label="Spark CART fit")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Synthetic training rows (base training rows repeated)")
    ax.set_ylabel("Median fit time (seconds, log scale)")
    if frame["fit_seconds_median"].notna().any():
        ax.set_title("Local vs Spark Decision Tree: synthetic scale test")
    else:
        ax.set_title("Local CART fit on synthetic replicated rows (Spark blocked)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "scaled_tree_benchmark.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved benchmark evidence under {output}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args()
    result = run(tuple(args.sizes), repeats=args.repeats)
    print(json.dumps({
        "status": result["status"],
        "startup_seconds": result["spark_startup_seconds_separate"],
        "sizes": [entry["synthetic_training_rows"] for entry in result["measurements"]],
        "output": str(PROJECT_DIR / "artifacts" / "spark"),
    }, indent=2))


if __name__ == "__main__":
    main()
