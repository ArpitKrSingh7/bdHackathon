#!/usr/bin/env python3
"""
scripts/grouped_cv_decision_tree.py
------------------------------------
Grouped cross-validation for the Decision Tree classifier on UCI Internet Ads.

Design:
  - Identical feature signatures stay in one fold (GroupKFold).
  - Imputation is fit inside each training fold — no leakage.
  - Reports class-aware measures because ads are ~14% of rows.
  - Does NOT overwrite the existing holdout results.
  - Saves additional CV metrics separately under artifacts/models/grouped_cv/.

Usage:
    python3 scripts/grouped_cv_decision_tree.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

# ── Project imports ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hackathon_utils import feature_signatures, load_ads

# ── Constants (match classification_clustering.py) ──
RANDOM_STATE = 42
N_FOLDS = 5
N_REPEATS = 3  # repeat the entire GroupKFold with different group shuffles
TREE_PARAMETERS = {
    "criterion": "gini",
    "max_depth": 5,
    "min_samples_leaf": 20,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "models" / "grouped_cv"


def _package_versions() -> dict:
    import sklearn
    from importlib.metadata import PackageNotFoundError, version
    result = {"python": platform.python_version(), "scikit-learn": sklearn.__version__}
    for pkg in ("pandas", "numpy", "scipy"):
        try:
            result[pkg] = version(pkg)
        except PackageNotFoundError:
            result[pkg] = "not installed"
    return result


def run_grouped_cv():
    """Execute repeated GroupKFold cross-validation and save per-fold + aggregated metrics."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    data, feature_names, numeric_features, binary_features = load_ads()
    y = (data["class"].astype(str).to_numpy() == "ad").astype(np.int8)
    X = data[feature_names]
    groups = feature_signatures(data, feature_names).to_numpy()

    # Verify groups: unique groups count
    unique_groups = np.unique(groups)
    n_unique_groups = len(unique_groups)
    print(f"Dataset: {len(data)} rows, {n_unique_groups} unique feature-signature groups")
    print(f"Class distribution: {int(y.sum())} ad ({y.mean()*100:.2f}%), "
          f"{int((y==0).sum())} nonad ({(y==0).mean()*100:.2f}%)")
    print(f"Tree parameters: {TREE_PARAMETERS}")
    print(f"Cross-validation: {N_FOLDS}-fold GroupKFold × {N_REPEATS} repeats = "
          f"{N_FOLDS * N_REPEATS} total folds")

    # Build preprocessor template
    def make_preprocessor():
        return ColumnTransformer(
            transformers=[
                ("geometry", SimpleImputer(strategy="median"), numeric_features),
                ("binary", SimpleImputer(strategy="most_frequent"), binary_features),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    fold_results = []
    overall_start = time.perf_counter()

    for repeat_idx in range(N_REPEATS):
        # Shuffle groups deterministically for each repeat
        rng = np.random.default_rng(RANDOM_STATE + repeat_idx)
        # Create a permuted group mapping so GroupKFold splits differently
        # GroupKFold splits by group identity, so we need to relabel groups
        shuffled_group_ids = np.arange(n_unique_groups)
        rng.shuffle(shuffled_group_ids)
        group_map = dict(zip(unique_groups, shuffled_group_ids))
        remapped_groups = np.array([group_map[g] for g in groups])

        gkf = GroupKFold(n_splits=N_FOLDS)

        for fold_idx, (train_pos, test_pos) in enumerate(
            gkf.split(X, y, groups=remapped_groups)
        ):
            # Verify no group leakage
            train_groups_set = set(remapped_groups[train_pos])
            test_groups_set = set(remapped_groups[test_pos])
            overlap = train_groups_set & test_groups_set
            if overlap:
                raise RuntimeError(
                    f"Group leakage in repeat {repeat_idx}, fold {fold_idx}: "
                    f"{len(overlap)} shared groups"
                )

            X_train_raw = X.iloc[train_pos]
            X_test_raw = X.iloc[test_pos]
            y_train = y[train_pos]
            y_test = y[test_pos]

            # Fit imputer on training fold only
            preprocessor = make_preprocessor()
            preprocessor.fit(X_train_raw)
            X_train = np.asarray(preprocessor.transform(X_train_raw), dtype=np.float64)
            X_test = np.asarray(preprocessor.transform(X_test_raw), dtype=np.float64)

            # Fit and predict
            model = DecisionTreeClassifier(**TREE_PARAMETERS)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test).astype(np.int8)
            positive_index = list(model.classes_).index(1)
            y_score = model.predict_proba(X_test)[:, positive_index]

            # Compute class-aware metrics
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            fold_metrics = {
                "repeat": repeat_idx,
                "fold": fold_idx,
                "n_train": int(len(train_pos)),
                "n_test": int(len(test_pos)),
                "train_ad": int(y_train.sum()),
                "test_ad": int(y_test.sum()),
                "train_ad_pct": float(y_train.mean() * 100),
                "test_ad_pct": float(y_test.mean() * 100),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "precision_ad": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
                "recall_ad": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
                "f1_ad": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, y_score)),
                "confusion_matrix_TN_FP_FN_TP": [int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])],
                "group_overlap": 0,
                "imputation_fitted_on": "training_fold_only",
            }
            fold_results.append(fold_metrics)

            print(f"  Repeat {repeat_idx}, Fold {fold_idx}: "
                  f"acc={fold_metrics['accuracy']:.4f}, "
                  f"bal_acc={fold_metrics['balanced_accuracy']:.4f}, "
                  f"ad_F1={fold_metrics['f1_ad']:.4f}, "
                  f"AUC={fold_metrics['roc_auc']:.4f}, "
                  f"train={len(train_pos)}/test={len(test_pos)}")

    total_time = time.perf_counter() - overall_start
    print(f"\nTotal CV time: {total_time:.4f} s")

    # Aggregate metrics
    metric_keys = ["accuracy", "balanced_accuracy", "precision_ad", "recall_ad", "f1_ad", "roc_auc"]
    aggregated = {}
    for key in metric_keys:
        values = [f[key] for f in fold_results]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "values": values,
        }

    # Print summary
    print("\n" + "=" * 70)
    print(f"Grouped CV Summary ({N_FOLDS}-fold × {N_REPEATS} repeats = {len(fold_results)} folds)")
    print("=" * 70)
    for key in metric_keys:
        agg = aggregated[key]
        print(f"  {key:25s}: {agg['mean']:.4f} ± {agg['std']:.4f}  "
              f"[{agg['min']:.4f}, {agg['max']:.4f}]")

    # Build full output
    output = {
        "method": f"{N_FOLDS}-fold GroupKFold × {N_REPEATS} repeats",
        "description": (
            "Repeated group-aware cross-validation. Identical feature signatures "
            "stay in one partition; imputation is fit inside each training fold. "
            "This supplements the existing holdout result, which is not modified."
        ),
        "tree_parameters": TREE_PARAMETERS,
        "n_folds": N_FOLDS,
        "n_repeats": N_REPEATS,
        "total_folds_evaluated": len(fold_results),
        "dataset_rows": len(data),
        "unique_signature_groups": int(n_unique_groups),
        "class_counts": {"ad": int(y.sum()), "nonad": int((y == 0).sum())},
        "total_cv_seconds": float(total_time),
        "environment": _package_versions(),
        "aggregated_metrics": aggregated,
        "per_fold_results": fold_results,
        "group_leakage_check": "verified zero overlap in every fold",
        "imputation_leakage_check": "imputer fit on training fold only in every fold",
    }

    # Save outputs
    output_path = OUTPUT_DIR / "grouped_cv_metrics.json"
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved to {output_path}")

    # Also save a compact CSV of per-fold metrics
    csv_path = OUTPUT_DIR / "grouped_cv_per_fold.csv"
    pd.DataFrame(fold_results).to_csv(csv_path, index=False)
    print(f"Saved per-fold CSV to {csv_path}")

    return output


if __name__ == "__main__":
    run_grouped_cv()
