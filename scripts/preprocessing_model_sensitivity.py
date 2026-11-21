#!/usr/bin/env python3
"""Evidence for two defensible preprocessing improvements.

1) Fit a univariate ANOVA feature selector on the training partition only and
   compare its held-out Decision Tree result with the all-feature baseline.
2) Create a separate audit variant excluding complete full-feature signatures
   whose labels conflict. The raw data and the main normalized dataset remain
   unchanged; this target-informed variant is for sensitivity analysis only.

IQR flags are reported but not automatically deleted: the raw audit found no
invalid geometry values, and a statistical fence alone does not establish noise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
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
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from classification_clustering import TREE_PARAMETERS, make_grouped_split  # noqa: E402
from hackathon_utils import feature_signatures, load_ads  # noqa: E402

SELECT_K = 100


def _score(y_true: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision_ad": float(precision_score(y_true, prediction, zero_division=0)),
        "recall_ad": float(recall_score(y_true, prediction, zero_division=0)),
        "f1_ad": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "confusion_matrix_nonad_ad": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
    }


def _fit_variant(
    data: pd.DataFrame,
    feature_names: list[str],
    numeric_features: list[str],
    binary_features: list[str],
    select_k: int | None,
) -> dict:
    train_pos, test_pos, split_metadata = make_grouped_split(data, feature_names)
    X_train_raw = data.iloc[train_pos][feature_names]
    X_test_raw = data.iloc[test_pos][feature_names]
    y = (data["class"].astype(str).to_numpy() == "ad").astype(np.int8)
    y_train, y_test = y[train_pos], y[test_pos]

    preprocessor = ColumnTransformer(
        transformers=[
            ("geometry", SimpleImputer(strategy="median"), numeric_features),
            ("binary", SimpleImputer(strategy="most_frequent"), binary_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    transformed_names = np.asarray(preprocessor.get_feature_names_out(), dtype=object)

    selector = None
    feature_scores = None
    candidate_feature_count = len(transformed_names)
    if select_k is not None:
        # A zero-variance filter is fitted on train only. Constant training
        # columns carry no class-separation information and can make ANOVA F
        # scores undefined.
        if hasattr(X_train, "toarray"):
            train_variance = np.asarray(X_train.power(2).mean(axis=0)).ravel() - np.square(
                np.asarray(X_train.mean(axis=0)).ravel()
            )
        else:
            train_variance = np.var(np.asarray(X_train), axis=0)
        variable_mask = np.isfinite(train_variance) & (train_variance > 0)
        X_train = X_train[:, variable_mask]
        X_test = X_test[:, variable_mask]
        transformed_names = transformed_names[variable_mask]
        candidate_feature_count = len(transformed_names)
        k = min(int(select_k), X_train.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
        raw_scores = np.asarray(selector.scores_, dtype=np.float64)
        raw_pvalues = np.asarray(selector.pvalues_, dtype=np.float64)
        raw_scores = np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0)
        raw_pvalues = np.nan_to_num(raw_pvalues, nan=1.0, posinf=1.0, neginf=1.0)
        order = np.argsort(-raw_scores, kind="stable")
        feature_scores = [
            {
                "feature": str(transformed_names[index]),
                "f_score": float(raw_scores[index]),
                "p_value": float(raw_pvalues[index]),
                "selected": bool(selector.get_support()[index]),
                "fit_partition": "training only",
            }
            for index in order
        ]
        selected_names = transformed_names[selector.get_support()].tolist()
    else:
        selected_names = transformed_names.tolist()

    model = DecisionTreeClassifier(**TREE_PARAMETERS)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test).astype(np.int8)
    probability = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    return {
        "selected_feature_count": len(selected_names),
        "all_feature_count": len(feature_names),
        "candidate_feature_count_after_train_only_zero_variance_filter": candidate_feature_count,
        "selected_features": selected_names,
        "feature_scores": feature_scores,
        "train_rows": len(train_pos),
        "test_rows": len(test_pos),
        "train_ad_rows": int(y_train.sum()),
        "test_ad_rows": int(y_test.sum()),
        "metrics": _score(y_test, prediction, probability),
        "split": {
            key: value for key, value in split_metadata.items()
            if key not in {"train_positions", "test_positions"}
        },
        "selection": (
            "SelectKBest(f_classif, k=%d), fit only on the training partition" % len(selected_names)
            if selector is not None else "No feature selector; all declared inputs retained"
        ),
    }


def _conflict_variant(data: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, dict]:
    signatures = feature_signatures(data, feature_names)
    frame = data.copy()
    frame["_signature"] = signatures.to_numpy()
    group_labels = frame.groupby("_signature", sort=False)["class"].nunique()
    conflict_signatures = set(group_labels[group_labels > 1].index)
    mask = frame["_signature"].isin(conflict_signatures)
    details = []
    for signature in sorted(conflict_signatures, key=str):
        group = frame.loc[frame["_signature"] == signature, "class"].value_counts()
        details.append({
            "feature_signature": str(signature),
            "rows": int(group.sum()),
            "class_counts": {str(label): int(count) for label, count in group.items()},
        })
    cleaned = data.loc[~mask.to_numpy()].copy().reset_index(drop=True)
    payload = {
        "method": "Exclude all rows from a full-feature signature group when that exact signature has both target labels.",
        "conflict_groups": len(conflict_signatures),
        "rows_excluded": int(mask.sum()),
        "class_counts_before": {str(k): int(v) for k, v in data["class"].value_counts().items()},
        "class_counts_after": {str(k): int(v) for k, v in cleaned["class"].value_counts().items()},
        "groups": details,
        "warning": "This is a retrospective, target-informed cleaning sensitivity variant. It is not used for the primary holdout metrics because screening labels across the full file can change the evaluation population.",
    }
    return cleaned, payload


def _iqr_flags(data: pd.DataFrame, numeric_features: list[str]) -> dict:
    flags = pd.DataFrame(False, index=data.index, columns=numeric_features)
    details = {}
    for feature in numeric_features:
        values = data[feature].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        spread = q3 - q1
        low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
        flags[feature] = data[feature].notna() & ((data[feature] < low) | (data[feature] > high))
        details[feature] = {
            "lower_fence": float(low),
            "upper_fence": float(high),
            "flagged_rows": int(flags[feature].sum()),
        }
    any_flag = flags.any(axis=1)
    return {
        "per_feature": details,
        "rows_flagged_any": int(any_flag.sum()),
        "rows_deleted": 0,
        "decision": "Retain these rows. IQR fences are review flags; the audit found finite positive geometry and the raw file does not establish that extreme formats are corrupt.",
    }


def main() -> None:
    data, feature_names, numeric_features, binary_features = load_ads()
    output = PROJECT_DIR / "artifacts" / "models" / "preprocessing_sensitivity"
    output.mkdir(parents=True, exist_ok=True)

    # Compare on the same grouped holdout, with selector statistics fitted on train only.
    baseline = _fit_variant(data, feature_names, numeric_features, binary_features, None)
    selected = _fit_variant(data, feature_names, numeric_features, binary_features, SELECT_K)
    pd.DataFrame(selected["feature_scores"]).to_csv(output / "anova_feature_scores.csv", index=False)

    top = pd.DataFrame(selected["feature_scores"]).head(20).sort_values("f_score")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["f_score"], color="#4b8f8c")
    ax.set_title("Top 20 training-only ANOVA feature scores")
    ax.set_xlabel("ANOVA F-score (larger indicates stronger univariate class separation)")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "anova_feature_scores_top20.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    conflict_data, conflict_report = _conflict_variant(data, feature_names)
    conflict_data.to_csv(
        PROJECT_DIR / "artifacts" / "processed" / "internet_ads_conflict_filtered_sensitivity.csv",
        index=False,
    )
    payload = {
        "dataset_rows": len(data),
        "feature_count": len(feature_names),
        "classification_feature_selection": {
            "method": "ANOVA F-statistic filter; k fixed at 100 before inspecting held-out metrics.",
            "selection_fit_on_training_partition_only": True,
            "baseline_all_features": baseline,
            "selected_top_100": selected,
            "interpretation_limit": "The held-out comparison is one grouped split. The selected model is a sensitivity analysis; K was not tuned on this holdout.",
        },
        "conflicting_label_cleaning_sensitivity": conflict_report,
        "geometry_iqr_review": _iqr_flags(data, numeric_features),
        "primary_dataset_unchanged": True,
    }
    (output / "preprocessing_sensitivity.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "baseline_features": baseline["selected_feature_count"],
        "selected_features": selected["selected_feature_count"],
        "baseline_metrics": baseline["metrics"],
        "selected_metrics": selected["metrics"],
        "conflict_groups": conflict_report["conflict_groups"],
        "ambiguous_rows_excluded_in_sensitivity_copy": conflict_report["rows_excluded"],
        "conflict_variant_class_counts": conflict_report["class_counts_after"],
        "iqr_flagged_rows_retained": payload["geometry_iqr_review"]["rows_flagged_any"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
