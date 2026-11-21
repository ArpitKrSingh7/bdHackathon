"""Matched local/PySpark benchmarks for the Internet Ads assignment.

Tasks:
  1. scikit-learn CART vs PySpark DecisionTreeClassifier on identical train/test
     rows and the same train-only, already-transformed feature matrices.
  2. local Apriori vs PySpark FP-Growth on identical transaction rows and
     support/confidence thresholds. These are different algorithms for the
     same frequent-itemset/rule-mining task, not identical implementations.

Spark startup is measured separately. If the runtime cannot launch Spark, the
script records the exact failure and produces no fabricated Spark task timings.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .hackathon_utils import RAW_DATA_DIR, load_ads
except ImportError:
    from hackathon_utils import RAW_DATA_DIR, load_ads

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts" / "spark"
ASSOCIATION_DIR = BASE_DIR / "artifacts" / "association"
MODEL_DIR = BASE_DIR / "artifacts" / "models"

try:
    from classification_clustering import TREE_PARAMETERS as TREE_PARAMS
except ImportError:
    try:
        from .classification_clustering import TREE_PARAMETERS as TREE_PARAMS
    except ImportError:
        TREE_PARAMS = {
            "criterion": "gini",
            "max_depth": 5,
            "min_samples_leaf": 20,
            "class_weight": "balanced",
            "random_state": 42,
        }
SPARK_TREE_PARAMS = {
    "impurity": "gini",
    "maxDepth": int(TREE_PARAMS.get("max_depth", 5)),
    "minInstancesPerNode": int(TREE_PARAMS.get("min_samples_leaf", 20)),
    "maxBins": 32,
    "seed": int(TREE_PARAMS.get("random_state", 42)),
    "weightCol": "class_weight",
}
ARM_PARAMS = {
    "min_support": 0.05,
    "min_confidence": 0.60,
    "min_item_support": 0.05,
    "num_partitions": 2,
}
SPARK_MASTER = os.environ.get("INTERNET_ADS_SPARK_MASTER", "local[2]")
REPEATS = 3


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metric_bundle(y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float]) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_ad": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_ad": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_ad": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc_ad": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) == 2 else None,
        "confusion_matrix_labels_0_nonad_1_ad": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def _encode_target(values: Sequence[Any]) -> np.ndarray:
    array = np.asarray(values)
    unique = set(array.tolist())
    if unique.issubset({"ad", "nonad"}):
        return np.asarray([1 if value == "ad" else 0 for value in array], dtype=np.int32)
    if unique.issubset({0, 1, 0.0, 1.0}):
        return array.astype(np.int32)
    raise ValueError(f"Expected target labels ad/nonad or 0/1; got {sorted(unique, key=str)}")


def _read_tree_arrays(npz_path: Path) -> dict:
    """Load prepared supervised split; expected arrays are train/test X and y.

    Accepted aliases allow the classification module to retain readable names while keeping
    the benchmark interface explicit. The preprocessing itself must be performed
    by that module on train rows only.
    """
    if not npz_path.is_file():
        raise FileNotFoundError(f"Prepared supervised split not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as stored:
        keys = set(stored.files)
        aliases = {
            "X_train": ("X_train", "x_train", "train_X"),
            "X_test": ("X_test", "x_test", "test_X"),
            "y_train": ("y_train", "train_y"),
            "y_test": ("y_test", "test_y"),
        }
        result = {}
        for canonical, candidates in aliases.items():
            key = next((candidate for candidate in candidates if candidate in keys), None)
            if key is None and canonical in {"train_positions", "test_positions", "train_class_weights", "test_class_weights"}:
                continue
            if key is None:
                raise KeyError(f"{npz_path} must contain {canonical}; available arrays={sorted(keys)}")
            result[canonical] = np.asarray(stored[key])
        for optional in (
            "train_positions", "test_positions", "train_class_weights", "test_class_weights",
        ):
            if optional in keys:
                result[optional] = np.asarray(stored[optional])
    result["X_train"] = np.asarray(result["X_train"], dtype=np.float64)
    result["X_test"] = np.asarray(result["X_test"], dtype=np.float64)
    result["y_train"] = _encode_target(result["y_train"])
    result["y_test"] = _encode_target(result["y_test"])
    for key in ("X_train", "X_test"):
        if not np.isfinite(result[key]).all():
            raise ValueError(f"{key} has non-finite values; Spark/local trees need the same train-only imputation first")
    if result["X_train"].shape[1] != result["X_test"].shape[1]:
        raise ValueError("Train/test feature dimensions differ")
    return result


def _sklearn_tree_benchmark(arrays: dict, repeats: int = REPEATS) -> dict:
    """Measure local CART on the exact matrix supplied to the Spark tree."""
    from sklearn.tree import DecisionTreeClassifier

    fit_times = []
    predict_times = []
    last_metrics = None
    for _ in range(repeats):
        model = DecisionTreeClassifier(**TREE_PARAMS)
        start = time.perf_counter()
        model.fit(arrays["X_train"], arrays["y_train"])
        fit_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        pred = model.predict(arrays["X_test"])
        score = model.predict_proba(arrays["X_test"])[:, 1]
        predict_times.append(time.perf_counter() - start)
        last_metrics = _metric_bundle(arrays["y_test"], pred, score)
    return {
        "engine": "scikit-learn",
        "parameters": TREE_PARAMS,
        "fit_seconds_repeats": fit_times,
        "fit_seconds_median": statistics.median(fit_times),
        "predict_seconds_repeats": predict_times,
        "predict_seconds_median": statistics.median(predict_times),
        "metrics": last_metrics,
        "train_rows": int(len(arrays["y_train"])),
        "test_rows": int(len(arrays["y_test"])),
        "features": int(arrays["X_train"].shape[1]),
        "input_note": "Same group-aware split and same train-only transformed matrices used for PySpark.",
    }


def _spark_tree_benchmark(spark, arrays: dict, repeats: int = REPEATS) -> dict:
    """Train PySpark CART with the same split, matrices, and balanced class weights."""
    from pyspark.ml.classification import DecisionTreeClassifier
    from pyspark.ml.linalg import VectorUDT, Vectors
    from pyspark.sql import Row
    from pyspark.sql.types import DoubleType, StructField, StructType

    counts = np.bincount(arrays["y_train"], minlength=2)
    if np.any(counts == 0):
        raise ValueError(f"Training split must contain both classes; counts={counts.tolist()}")
    total = int(counts.sum())
    row_weights = {class_id: total / (2.0 * int(count)) for class_id, count in enumerate(counts)}
    train_weights = arrays.get("train_class_weights")
    if train_weights is None:
        train_weights = np.asarray([row_weights[int(label)] for label in arrays["y_train"]], dtype=np.float64)
    else:
        train_weights = np.asarray(train_weights, dtype=np.float64)
        if len(train_weights) != len(arrays["y_train"]):
            raise ValueError("Prepared train_class_weights length does not match y_train")

    train_schema = StructType([
        StructField("features", VectorUDT(), nullable=False),
        StructField("label", DoubleType(), nullable=False),
        StructField("class_weight", DoubleType(), nullable=False),
    ])
    test_schema = StructType([
        StructField("features", VectorUDT(), nullable=False),
        StructField("label", DoubleType(), nullable=False),
    ])

    start = time.perf_counter()
    train_records = [
        Row(features=Vectors.dense(row), label=float(label), class_weight=float(weight))
        for row, label, weight in zip(arrays["X_train"], arrays["y_train"], train_weights)
    ]
    test_records = [
        Row(features=Vectors.dense(row), label=float(label))
        for row, label in zip(arrays["X_test"], arrays["y_test"])
    ]
    python_row_build_seconds = time.perf_counter() - start

    start = time.perf_counter()
    train_df = spark.createDataFrame(train_records, schema=train_schema).cache()
    test_df = spark.createDataFrame(test_records, schema=test_schema).cache()
    materialized_rows = (train_df.count(), test_df.count())
    dataframe_conversion_seconds = time.perf_counter() - start

    params = dict(SPARK_TREE_PARAMS)
    model_fit_times = []
    prediction_times = []
    last_metrics = None
    prediction_records = None
    for _ in range(repeats):
        model = DecisionTreeClassifier(**params)
        start = time.perf_counter()
        fitted = model.fit(train_df)
        model_fit_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        rows = fitted.transform(test_df).select("label", "prediction", "probability").collect()
        prediction_times.append(time.perf_counter() - start)
        y_true = np.asarray([int(row.label) for row in rows], dtype=np.int32)
        y_pred = np.asarray([int(row.prediction) for row in rows], dtype=np.int32)
        y_score = np.asarray([float(row.probability[1]) for row in rows], dtype=np.float64)
        last_metrics = _metric_bundle(y_true, y_pred, y_score)
        prediction_records = rows

    return {
        "engine": "pyspark.ml DecisionTreeClassifier",
        "parameters": params,
        "class_weights": {"nonad_0": row_weights[0], "ad_1": row_weights[1]},
        "python_row_build_seconds": python_row_build_seconds,
        "spark_dataframe_conversion_and_materialization_seconds": dataframe_conversion_seconds,
        "fit_seconds_repeats": model_fit_times,
        "fit_seconds_median": statistics.median(model_fit_times),
        "predict_and_collect_seconds_repeats": prediction_times,
        "predict_and_collect_seconds_median": statistics.median(prediction_times),
        "metrics": last_metrics,
        "materialized_rows": list(materialized_rows),
        "train_rows": int(len(arrays["y_train"])),
        "test_rows": int(len(arrays["y_test"])),
        "features": int(arrays["X_train"].shape[1]),
        "input_note": "Same group-aware split and same train-only transformed matrices as local CART; Spark maxBins=32 and sklearn exact-threshold search may differ.",
    }


def _save_frame(frame: pd.DataFrame, filename: str) -> Optional[str]:
    if frame is None:
        return None
    path = ARTIFACT_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return str(path)


def _normalize_itemset_frame(frame: pd.DataFrame, items_column: str = "items") -> pd.DataFrame:
    result = frame.copy()
    if items_column in result.columns:
        result[items_column] = result[items_column].map(lambda xs: " | ".join(sorted(map(str, xs))))
    for col in ("antecedent", "consequent"):
        if col in result.columns:
            result[col] = result[col].map(lambda xs: " | ".join(sorted(map(str, xs))))
    return result


def _run_local_apriori(data, binary_features) -> tuple[dict, dict]:
    """Call the local association baseline using the shared interface."""
    try:
        from association_mining import build_transactions, run_apriori
    except ImportError:
        from .association_mining import build_transactions, run_apriori

    transaction_data = build_transactions(
        data,
        binary_features,
        include_class=True,
        min_item_support=ARM_PARAMS["min_item_support"],
        target_col="class",
    )
    result = run_apriori(
        transaction_data,
        min_support=ARM_PARAMS["min_support"],
        min_confidence=ARM_PARAMS["min_confidence"],
    )
    itemsets = result.get("itemsets")
    all_rules = result.get("rules")
    rules = result.get("class_rules", all_rules)
    if not isinstance(itemsets, pd.DataFrame):
        raise TypeError("association_mining.run_apriori must return a pandas DataFrame under 'itemsets'")
    if not isinstance(all_rules, pd.DataFrame):
        all_rules = pd.DataFrame(all_rules)
    if not isinstance(rules, pd.DataFrame):
        rules = pd.DataFrame(rules)
    local = {
        "engine": "local Apriori",
        "parameters": ARM_PARAMS,
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "transaction_count": int(result["transaction_count"]),
        "selected_features": int(len(transaction_data["selected_features"])),
        "frequent_itemset_count": int(len(itemsets)),
        "association_rule_count": int(len(all_rules)),
        "class_rule_count": int(len(rules)),
    }
    _save_frame(_normalize_itemset_frame(itemsets), "local_apriori_itemsets.csv")
    _save_frame(_normalize_itemset_frame(rules), "local_apriori_class_rules.csv")
    return transaction_data, {"summary": local, "itemsets": itemsets, "rules": rules}


def _spark_fpgrowth_benchmark(spark, transaction_data: dict, repeats: int = REPEATS) -> dict:
    """Run Spark FP-Growth on the exact row transactions used by local Apriori."""
    from pyspark.ml.fpm import FPGrowth
    from pyspark.sql.types import ArrayType, StringType, StructField, StructType

    transactions = transaction_data["transactions"]
    schema = StructType([StructField("items", ArrayType(StringType(), containsNull=False), nullable=False)])
    start = time.perf_counter()
    spark_rows = [(list(items),) for items in transactions]
    python_row_build_seconds = time.perf_counter() - start
    start = time.perf_counter()
    transaction_df = spark.createDataFrame(spark_rows, schema=schema).cache()
    row_count = transaction_df.count()
    dataframe_conversion_seconds = time.perf_counter() - start
    if row_count != len(transactions):
        raise ValueError(f"Spark transaction count {row_count} != local count {len(transactions)}")

    fit_times = []
    collect_times = []
    final_itemsets = None
    final_rules = None
    model = None
    for _ in range(repeats):
        estimator = FPGrowth(
            itemsCol="items",
            minSupport=ARM_PARAMS["min_support"],
            minConfidence=ARM_PARAMS["min_confidence"],
            numPartitions=ARM_PARAMS["num_partitions"],
        )
        start = time.perf_counter()
        model = estimator.fit(transaction_df)
        fit_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        itemset_rows = model.freqItemsets.collect()
        rule_rows = model.associationRules.collect()
        collect_times.append(time.perf_counter() - start)
        final_itemsets = pd.DataFrame([row.asDict(recursive=True) for row in itemset_rows])
        final_rules = pd.DataFrame([row.asDict(recursive=True) for row in rule_rows])

    _save_frame(_normalize_itemset_frame(final_itemsets), "spark_fpgrowth_itemsets.csv")
    _save_frame(_normalize_itemset_frame(final_rules), "spark_fpgrowth_rules.csv")
    return {
        "engine": "pyspark.ml.fpm.FPGrowth",
        "parameters": ARM_PARAMS,
        "python_row_build_seconds": python_row_build_seconds,
        "spark_dataframe_conversion_and_materialization_seconds": dataframe_conversion_seconds,
        "fit_seconds_repeats": fit_times,
        "fit_seconds_median": statistics.median(fit_times),
        "collect_itemsets_and_rules_seconds_repeats": collect_times,
        "collect_itemsets_and_rules_seconds_median": statistics.median(collect_times),
        "transaction_count": int(row_count),
        "frequent_itemset_count": int(len(final_itemsets)),
        "association_rule_count": int(len(final_rules)),
        "selected_features": int(len(transaction_data["selected_features"])),
        "itemsets_artifact": str(ARTIFACT_DIR / "spark_fpgrowth_itemsets.csv"),
        "rules_artifact": str(ARTIFACT_DIR / "spark_fpgrowth_rules.csv"),
        "algorithm_note": "Spark FP-Growth and local Apriori differ algorithmically; this is a task-level runtime comparison on the same transactions and thresholds.",
    }


def _load_prepared_tree_data() -> tuple[dict, Path]:
    """Read classification module artifact. Filename can be overridden by env."""
    configured = os.environ.get("INTERNET_ADS_TREE_ARRAYS")
    candidates = [Path(configured)] if configured else []
    candidates.extend([
        MODEL_DIR / "tree_shared_split.npz",
        MODEL_DIR / "tree_split_arrays.npz",
        MODEL_DIR / "supervised_split.npz",
        MODEL_DIR / "decision_tree_split.npz",
    ])
    artifact = next((path for path in candidates if path.is_file()), None)
    if artifact is None:
        raise FileNotFoundError(
            "Could not find prepared classification NPZ. Set INTERNET_ADS_TREE_ARRAYS "
            "or save train/test X and y arrays in artifacts/models/tree_split_arrays.npz."
        )
    prepared = _read_tree_arrays(artifact)
    metadata_path = MODEL_DIR / "tree_split_metadata.json"
    if metadata_path.is_file():
        prepared["split_metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
    return prepared, artifact


def run_benchmarks() -> dict:
    """Run the local baselines and two Spark tasks, recording only real timings."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "started",
        "spark_master": SPARK_MASTER,
        "pyspark_version": None,
        "python_version": sys.version,
        "parameters": {"tree_local": TREE_PARAMS, "tree_spark": SPARK_TREE_PARAMS, "arm": ARM_PARAMS},
        "hardware": {"available_cpu_count": os.cpu_count()},
    }
    try:
        import pyspark
        summary["pyspark_version"] = pyspark.__version__
    except Exception as exc:
        summary["pyspark_import_error"] = f"{type(exc).__name__}: {exc}"

    # Local preprocessing and task baselines are performed before the Spark startup
    # attempt. These timings are useful even when sandbox policy blocks Py4J sockets.
    data, feature_names, _, binary_features = load_ads(RAW_DATA_DIR)
    local_arm = None
    transaction_data = None
    try:
        transaction_data, local_arm = _run_local_apriori(data, binary_features)
        summary["local_apriori"] = local_arm["summary"]
    except Exception as exc:
        summary["local_apriori_error"] = f"{type(exc).__name__}: {exc}"

    arrays = None
    local_tree = None
    tree_artifact = None
    try:
        arrays, tree_artifact = _load_prepared_tree_data()
        local_tree = _sklearn_tree_benchmark(arrays)
        summary["local_tree"] = local_tree
        summary["tree_arrays_artifact"] = str(tree_artifact)
        split_meta = arrays.get("split_metadata", {})
        summary["tree_split_summary"] = {
            key: split_meta.get(key)
            for key in (
                "test_size_observed", "class_prevalence_full", "class_prevalence_train",
                "class_prevalence_test", "random_state", "candidate_splits_considered",
                "train_signature_groups", "test_signature_groups", "signature_overlap",
            )
        } if split_meta else None
        summary["tree_split_metadata_artifact"] = str(MODEL_DIR / "tree_split_metadata.json")
    except Exception as exc:
        summary["local_tree_error"] = f"{type(exc).__name__}: {exc}"

    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    start = time.perf_counter()
    spark = None
    try:
        from pyspark.sql import SparkSession
        spark = (
            SparkSession.builder
            .master(SPARK_MASTER)
            .appName("internet-ads-matched-benchmark")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.sql.shuffle.partitions", str(ARM_PARAMS["num_partitions"]))
            .config("spark.default.parallelism", str(ARM_PARAMS["num_partitions"]))
            .getOrCreate()
        )
        summary["spark_startup_seconds"] = time.perf_counter() - start
        summary["spark_initialization_completed"] = True
        summary["spark_version"] = spark.version
        spark.sparkContext.setLogLevel("ERROR")
        if arrays is not None:
            summary["spark_tree"] = _spark_tree_benchmark(spark, arrays)
        else:
            summary["spark_tree_skipped"] = "Prepared train/test matrices were not available."
        if transaction_data is not None:
            summary["spark_fpgrowth"] = _spark_fpgrowth_benchmark(spark, transaction_data)
        else:
            summary["spark_fpgrowth_skipped"] = "Local transaction builder/APRIORI result was unavailable."
        summary["status"] = "completed"
    except Exception as exc:
        summary["spark_startup_attempt_seconds"] = time.perf_counter() - start
        summary["spark_initialization_completed"] = False
        summary["status"] = "spark_failed"
        summary["spark_error_type"] = type(exc).__name__
        summary["spark_error"] = str(exc)
        summary["spark_error_detail"] = repr(exc)
        summary["spark_tasks_completed"] = False
    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception as exc:
                summary["spark_stop_warning"] = f"{type(exc).__name__}: {exc}"

    if "spark_tree" in summary and "local_tree" in summary:
        summary["tree_timing_comparison"] = {
            "local_fit_seconds_median": local_tree["fit_seconds_median"],
            "spark_fit_seconds_median": summary["spark_tree"]["fit_seconds_median"],
            "spark_fit_over_local_ratio": summary["spark_tree"]["fit_seconds_median"] / local_tree["fit_seconds_median"],
            "spark_end_to_end_excluding_startup_seconds": (
                summary["spark_tree"]["python_row_build_seconds"]
                + summary["spark_tree"]["spark_dataframe_conversion_and_materialization_seconds"]
                + summary["spark_tree"]["fit_seconds_median"]
                + summary["spark_tree"]["predict_and_collect_seconds_median"]
            ),
        }
    if "spark_fpgrowth" in summary and "local_apriori" in summary:
        spark_fp = summary["spark_fpgrowth"]
        summary["arm_timing_comparison"] = {
            "local_apriori_seconds": summary["local_apriori"]["elapsed_seconds"],
            "spark_fpgrowth_fit_seconds_median": spark_fp["fit_seconds_median"],
            "spark_fp_end_to_end_excluding_startup_seconds": (
                spark_fp["python_row_build_seconds"]
                + spark_fp["spark_dataframe_conversion_and_materialization_seconds"]
                + spark_fp["fit_seconds_median"]
                + spark_fp["collect_itemsets_and_rules_seconds_median"]
            ),
            "algorithm_comparison": "Different frequent-pattern algorithms; compare task outputs and runtime, not implementation equivalence.",
        }

    summary["dataset"] = {
        "rows": int(len(data)),
        "columns_with_target": int(data.shape[1]),
        "input_features": int(len(feature_names)),
        "binary_features": int(len(binary_features)),
        "class_counts": {str(key): int(value) for key, value in data["class"].value_counts().items()},
        "repeated_rows_preserved": True,
    }
    summary["runtime_environment"] = {
        "python": sys.version.split()[0],
        "cpu_count_visible": os.cpu_count(),
        "spark_master": SPARK_MASTER,
        "spark_ui_enabled": False,
    }
    _write_json(ARTIFACT_DIR / "pyspark_benchmark_summary.json", summary)
    return summary


def main() -> int:
    try:
        result = run_benchmarks()
    except Exception as exc:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "failed_before_benchmark",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "error_detail": repr(exc),
        }
        _write_json(ARTIFACT_DIR / "pyspark_benchmark_summary.json", failure)
        print(json.dumps(failure, indent=2))
        return 1
    print(json.dumps(_json_safe(result), indent=2))
    return 0 if result.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
