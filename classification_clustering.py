"""Decision Tree and DBSCAN analysis for the UCI Internet Ads assignment.

This module preserves repeated rows. The classification holdout is split by the
full input-feature signature, so an identical feature vector cannot cross from
train to test. All supervised imputation is fit on the training rows only.
"""
from __future__ import annotations

import json
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from hackathon_utils import RAW_DATA_DIR, feature_signatures, load_ads, raw_file_manifest

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20
TREE_PARAMETERS = {
    "criterion": "gini",
    "max_depth": 5,
    "min_samples_leaf": 20,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}
DBSCAN_MIN_SAMPLES = 10
DBSCAN_MIN_SAMPLES_SENSITIVITY = (6, 10, 15)
EPS_MULTIPLIERS = (0.75, 1.0, 1.25, 1.5)


def _package_versions() -> dict:
    result = {"python": platform.python_version(), "scikit-learn": sklearn.__version__}
    for package in ("pandas", "numpy", "matplotlib", "scipy"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not installed"
    return result


def make_grouped_split(
    data: pd.DataFrame,
    feature_names: List[str],
    target_col: str = "class",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    n_candidates: int = 300,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Return deterministic positional indices with duplicate signatures grouped.

    GroupShuffleSplit does not stratify. We evaluate deterministic candidate group
    splits and choose the candidate closest to requested test size and full-data ad
    prevalence. Targets are used only to balance the holdout, never as predictors.
    """
    y = (data[target_col].astype(str).to_numpy() == "ad").astype(np.int8)
    groups = feature_signatures(data, feature_names).to_numpy()
    splitter = GroupShuffleSplit(
        n_splits=n_candidates, test_size=test_size, random_state=random_state
    )
    global_rate = float(y.mean())
    best = None
    for train_positions, test_positions in splitter.split(
        np.zeros((len(data), 1)), y, groups
    ):
        train_rate = float(y[train_positions].mean())
        test_rate = float(y[test_positions].mean())
        test_fraction = len(test_positions) / len(data)
        if min(y[train_positions].sum(), len(train_positions) - y[train_positions].sum(),
               y[test_positions].sum(), len(test_positions) - y[test_positions].sum()) == 0:
            continue
        score = (
            abs(test_fraction - test_size)
            + abs(test_rate - global_rate)
            + 0.25 * abs(train_rate - global_rate)
        )
        if best is None or score < best[0]:
            best = (score, train_positions.copy(), test_positions.copy())
    if best is None:
        raise ValueError("Could not find a group-aware split containing both classes")

    score, train_positions, test_positions = best
    train_groups = set(groups[train_positions])
    test_groups = set(groups[test_positions])
    if train_groups.intersection(test_groups):
        raise RuntimeError("A duplicate feature signature crossed the holdout split")
    metadata = {
        "test_size_requested": test_size,
        "test_size_observed": len(test_positions) / len(data),
        "class_prevalence_full": global_rate,
        "class_prevalence_train": float(y[train_positions].mean()),
        "class_prevalence_test": float(y[test_positions].mean()),
        "candidate_split_score": float(score),
        "random_state": random_state,
        "candidate_splits_considered": n_candidates,
        "train_signature_groups": len(train_groups),
        "test_signature_groups": len(test_groups),
        "signature_overlap": 0,
        "train_positions": train_positions.tolist(),
        "test_positions": test_positions.tolist(),
    }
    return train_positions, test_positions, metadata


def _extract_if_then_rules(model: DecisionTreeClassifier, feature_names: List[str]) -> List[dict]:
    """Walk sklearn's fitted CART and return a readable IF/THEN rule per leaf."""
    tree = model.tree_
    rules: List[dict] = []

    def visit(node: int, conditions: List[str]) -> None:
        left = tree.children_left[node]
        right = tree.children_right[node]
        if left == right:
            class_index = int(np.argmax(tree.value[node][0]))
            predicted = int(model.classes_[class_index])
            predicted_class_probability = float(
                tree.value[node][0][class_index] / tree.value[node][0].sum()
            )
            antecedent = " AND ".join(conditions) if conditions else "TRUE"
            rules.append({
                "rule_id": len(rules) + 1,
                "if": antecedent,
                "then": "ad" if predicted == 1 else "nonad",
                "training_rows_at_leaf": int(tree.n_node_samples[node]),
                "weighted_predicted_class_probability_at_leaf": predicted_class_probability,
                "text": f"IF {antecedent} THEN class={'ad' if predicted == 1 else 'nonad'}",
            })
            return
        feature_index = int(tree.feature[node])
        feature = feature_names[feature_index]
        threshold = float(tree.threshold[node])
        if feature in {"ad", "nonad", "class"}:
            raise RuntimeError("Target feature found in tree rules")
        if feature_index >= 3:
            left_condition = f"{feature} = 0"
            right_condition = f"{feature} = 1"
        else:
            left_condition = f"{feature} <= {threshold:.6g}"
            right_condition = f"{feature} > {threshold:.6g}"
        visit(left, conditions + [left_condition])
        visit(right, conditions + [right_condition])

    visit(0, [])
    return rules


def _save_tree_figures(
    model: DecisionTreeClassifier,
    feature_names: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    output_dir: Path,
) -> dict:
    labels = ["nonad", "ad"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("Decision Tree — test confusion matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    threshold = max(cm.max() / 2, 1)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{cm[row, col]:,}", ha="center", va="center",
                    color="white" if cm[row, col] > threshold else "black")
    fig.colorbar(image, ax=ax, label="Test rows")
    fig.tight_layout()
    cm_path = output_dir / "decision_tree_confusion_matrix.png"
    fig.savefig(cm_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.plot(fpr, tpr, color="#bd4b3b", linewidth=2, label=f"Tree (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Chance")
    ax.set_title("Decision Tree — test ROC curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    roc_path = output_dir / "decision_tree_roc.png"
    fig.savefig(roc_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(21, 10))
    plot_tree(
        model,
        max_depth=TREE_PARAMETERS["max_depth"],
        feature_names=feature_names,
        class_names=labels,
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        fontsize=6,
        ax=ax,
    )
    ax.set_title("Decision Tree — fitted CART (maximum depth 5)")
    fig.tight_layout()
    tree_path = output_dir / "decision_tree_structure.png"
    fig.savefig(tree_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances[importances > 0].sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.sort_values().plot(kind="barh", ax=ax, color="#4b8f8c")
    ax.set_title("Decision Tree — most-used features")
    ax.set_xlabel("Total Gini importance")
    ax.set_ylabel("Input feature")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    importance_path = output_dir / "decision_tree_feature_importance.png"
    fig.savefig(importance_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "confusion_matrix": str(cm_path),
        "roc_curve": str(roc_path),
        "tree_structure": str(tree_path),
        "feature_importance": str(importance_path),
        "auc_from_roc": float(auc),
    }


def run_decision_tree(
    data: pd.DataFrame,
    numeric_features: List[str],
    binary_features: List[str],
    feature_names: List[str],
    output_dir: Path = ARTIFACT_DIR,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    y = (data["class"].astype(str).to_numpy() == "ad").astype(np.int8)
    train_pos, test_pos, split_metadata = make_grouped_split(data, feature_names)
    X = data[feature_names]
    X_train_raw = X.iloc[train_pos]
    X_test_raw = X.iloc[test_pos]
    y_train = y[train_pos]
    y_test = y[test_pos]

    # All imputation statistics are fit on train only, then reused for the holdout.
    preprocessor = ColumnTransformer(
        transformers=[
            ("geometry", SimpleImputer(strategy="median"), numeric_features),
            ("binary", SimpleImputer(strategy="most_frequent"), binary_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocess_start = time.perf_counter()
    preprocessor.fit(X_train_raw)
    X_train = np.asarray(preprocessor.transform(X_train_raw), dtype=np.float64)
    X_test = np.asarray(preprocessor.transform(X_test_raw), dtype=np.float64)
    preprocessing_seconds = time.perf_counter() - preprocess_start
    transformed_names = preprocessor.get_feature_names_out().tolist()

    model = DecisionTreeClassifier(**TREE_PARAMETERS)
    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    y_pred = model.predict(X_test).astype(np.int8)
    positive_index = list(model.classes_).index(1)
    y_score = model.predict_proba(X_test)[:, positive_index]
    prediction_seconds = time.perf_counter() - predict_start

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision_ad": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall_ad": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1_ad": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "confusion_matrix_labels_nonad_ad": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, labels=[0, 1], target_names=["nonad", "ad"],
            output_dict=True, zero_division=0,
        ),
        "n_train": int(len(train_pos)),
        "n_test": int(len(test_pos)),
        "train_ad": int(y_train.sum()),
        "test_ad": int(y_test.sum()),
        "preprocessing_fit_transform_seconds": float(preprocessing_seconds),
        "tree_fit_seconds": float(fit_seconds),
        "prediction_seconds": float(prediction_seconds),
        "tree_parameters": TREE_PARAMETERS,
        "preprocessing": {
            "continuous": "training-partition median imputation",
            "binary": "training-partition most-frequent imputation",
            "scaling": "none (CART split thresholds do not require feature scaling)",
        },
        "split": {key: value for key, value in split_metadata.items()
                  if key not in ("train_positions", "test_positions")},
    }
    rules = _extract_if_then_rules(model, transformed_names)
    metrics["n_if_then_rules"] = len(rules)
    metrics["rule_probability_definition"] = (
        "weighted_predicted_class_probability_at_leaf is the share of the leaf's "
        "class-weighted training mass belonging to the class predicted at that leaf; "
        "it is not the unweighted fraction of ads."
    )
    figure_paths = _save_tree_figures(
        model, transformed_names, y_test, y_pred, y_score, output_dir
    )
    metrics["figures"] = figure_paths

    pd.DataFrame(rules).to_csv(output_dir / "decision_tree_if_then_rules.csv", index=False)
    pd.DataFrame({
        "row_position": test_pos,
        "feature_signature": feature_signatures(data, feature_names).to_numpy()[test_pos],
        "actual": np.where(y_test == 1, "ad", "nonad"),
        "predicted": np.where(y_pred == 1, "ad", "nonad"),
        "ad_probability": y_score,
    }).to_csv(output_dir / "decision_tree_test_predictions.csv", index=False)

    # Matrices and exact positional indices are shared with the PySpark agent so
    # the two Decision Trees can use the same grouped holdout and preprocessing.
    np.savez_compressed(
        output_dir / "tree_shared_split.npz",
        train_positions=train_pos,
        test_positions=test_pos,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_class_weights=np.where(y_train == 1,
                                    len(y_train)/(2*max(y_train.sum(), 1)),
                                    len(y_train)/(2*max((y_train == 0).sum(), 1))),
        test_class_weights=np.where(y_test == 1,
                                    len(y_train)/(2*max(y_train.sum(), 1)),
                                    len(y_train)/(2*max((y_train == 0).sum(), 1))),
    )
    (output_dir / "tree_feature_names.json").write_text(
        json.dumps(transformed_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "tree_split_metadata.json").write_text(
        json.dumps({
            **split_metadata,
            "tree_parameters": TREE_PARAMETERS,
            "preprocessing": metrics["preprocessing"],
            "metrics": metrics,
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "decision_tree_rules.txt").write_text(
        "# weighted_predicted_class_probability_at_leaf is the share of the leaf's "
        "class-weighted training mass belonging to the class predicted at that leaf.\n"
        + "\n".join(rule["text"] +
                 f"  [train_rows={rule['training_rows_at_leaf']}, "
                 f"weighted_predicted_class_probability_at_leaf="
                 f"{rule['weighted_predicted_class_probability_at_leaf']:.3f}]"
                 for rule in rules) + "\n",
        encoding="utf-8",
    )
    return {"metrics": metrics, "rules": rules, "split": split_metadata,
            "output_dir": str(output_dir)}


def _knee_from_k_distances(k_distances: np.ndarray) -> Tuple[float, int, np.ndarray]:
    ordered = np.sort(np.asarray(k_distances, dtype=np.float64))
    x = np.linspace(0.0, 1.0, len(ordered))
    y_range = float(np.ptp(ordered))
    if y_range == 0:
        raise ValueError("k-distance curve is constant; an elbow epsilon is not identifiable")
    y = (ordered - ordered[0]) / y_range
    # Maximum perpendicular distance from the endpoint chord is the chosen elbow.
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    distances = np.abs((y1-y0)*x - (x1-x0)*y + x1*y0 - y1*x0) / np.hypot(y1-y0, x1-x0)
    knee_index = int(np.argmax(distances))
    return float(ordered[knee_index]), knee_index, ordered


def _cluster_silhouette(X: np.ndarray, labels: np.ndarray, random_state: int) -> float | None:
    non_noise = labels != -1
    cluster_ids = np.unique(labels[non_noise])
    if len(cluster_ids) < 2 or int(non_noise.sum()) <= len(cluster_ids):
        return None
    positions = np.flatnonzero(non_noise)
    if len(positions) > 1200:
        rng = np.random.default_rng(random_state)
        positions = np.sort(rng.choice(positions, size=1200, replace=False))
    sampled_labels = labels[positions]
    if len(np.unique(sampled_labels)) < 2:
        return None
    return float(silhouette_score(X[positions], sampled_labels, metric="euclidean"))


def run_dbscan(
    data: pd.DataFrame,
    numeric_features: List[str],
    output_dir: Path = ARTIFACT_DIR,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(numeric_features) != 3:
        raise ValueError(f"Expected 3 geometry features; received {numeric_features}")
    raw_geometry = data[numeric_features].copy()
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(raw_geometry)
    if np.any(imputed < 0):
        raise ValueError("log1p geometry transform requires nonnegative inputs")
    log_geometry = np.log1p(imputed)
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    X_cluster = scaler.fit_transform(log_geometry)

    min_samples = DBSCAN_MIN_SAMPLES
    neighbors = NearestNeighbors(n_neighbors=min_samples, metric="euclidean")
    neighbors.fit(X_cluster)
    distances, _ = neighbors.kneighbors(X_cluster)
    # The point itself is included by nearest-neighbor query, matching DBSCAN's
    # convention that min_samples includes the point under consideration.
    k_distances = distances[:, -1]
    eps, knee_index, sorted_k_distances = _knee_from_k_distances(k_distances)
    x_sorted = np.linspace(0, len(sorted_k_distances) - 1, len(sorted_k_distances))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(x_sorted, sorted_k_distances, color="#375c9a", linewidth=1.8)
    ax.axhline(eps, color="#c43c39", linestyle="--", label=f"selected knee ε={eps:.4f}")
    ax.scatter([knee_index], [eps], color="#c43c39", zorder=3)
    ax.set_title(f"DBSCAN — sorted {min_samples}-NN distances")
    ax.set_xlabel("Points sorted by k-distance")
    ax.set_ylabel(f"Distance to {min_samples}th nearest neighbor (scaled Euclidean)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    kdist_path = output_dir / "dbscan_k_distance.png"
    fig.savefig(kdist_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    y_external = (data["class"].astype(str).to_numpy() == "ad").astype(np.int8)
    # Two separate sensitivity sweeps avoid coupling epsilon to min_samples:
    # (a) vary epsilon around the selected 10-NN elbow, holding min_samples=10;
    # (b) vary min_samples at the selected elbow epsilon.
    sweep_specs = []
    for multiplier in EPS_MULTIPLIERS:
        sweep_specs.append(("epsilon", float(eps * multiplier), min_samples, float(multiplier)))
    for candidate_min_samples in DBSCAN_MIN_SAMPLES_SENSITIVITY:
        sweep_specs.append(("min_samples", float(eps), candidate_min_samples, 1.0))

    rows = []
    fitted_labels = {}
    for sensitivity_axis, test_eps, candidate_min_samples, multiplier in sweep_specs:
        model = DBSCAN(eps=test_eps, min_samples=candidate_min_samples,
                       metric="euclidean", algorithm="auto")
        labels = model.fit_predict(X_cluster)
        clusters = sorted(set(labels) - {-1})
        noise_count = int(np.sum(labels == -1))
        silhouette = _cluster_silhouette(X_cluster, labels, RANDOM_STATE)
        non_noise = labels != -1
        ari_non_noise = (
            float(adjusted_rand_score(y_external[non_noise], labels[non_noise]))
            if non_noise.sum() > 0 else None
        )
        rows.append({
            "sensitivity_axis": sensitivity_axis,
            "eps": test_eps,
            "eps_multiplier": multiplier,
            "min_samples": candidate_min_samples,
            "n_clusters_excluding_noise": len(clusters),
            "noise_count": noise_count,
            "noise_fraction": noise_count / len(labels),
            "silhouette_non_noise_sampled": silhouette,
            "adjusted_rand_vs_class_all_rows_posthoc": float(adjusted_rand_score(y_external, labels)),
            "adjusted_rand_vs_class_non_noise_posthoc": ari_non_noise,
        })
        if sensitivity_axis == "epsilon" and candidate_min_samples == min_samples and np.isclose(multiplier, 1.0):
            fitted_labels["labels"] = labels
            fitted_labels["model"] = model
    sensitivity = pd.DataFrame(rows).sort_values(["sensitivity_axis", "min_samples", "eps_multiplier"])
    sensitivity.to_csv(output_dir / "dbscan_parameter_sensitivity.csv", index=False)
    if "labels" not in fitted_labels:
        raise RuntimeError("Selected DBSCAN setting was not present in sensitivity grid")
    labels = fitted_labels["labels"]
    cluster_ids = sorted(set(labels) - {-1})
    noise_count = int(np.sum(labels == -1))

    cluster_membership = pd.DataFrame({
        "row_position": np.arange(len(data)),
        "cluster": labels,
        "class_for_posthoc_profile_only": data["class"].astype(str).to_numpy(),
        **{f"raw_{feature}": data[feature].to_numpy() for feature in numeric_features},
    })
    cluster_membership.to_csv(output_dir / "dbscan_membership_and_noise.csv", index=False)
    noise = cluster_membership.loc[cluster_membership["cluster"] == -1]
    noise.to_csv(output_dir / "dbscan_noise_points.csv", index=False)
    profile = pd.crosstab(
        cluster_membership["cluster"],
        cluster_membership["class_for_posthoc_profile_only"],
        margins=True,
    )
    profile.to_csv(output_dir / "dbscan_cluster_class_profile.csv")

    # Plot raw geometry on log axes, but cluster colors come from transformed 3D data.
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    palette = plt.get_cmap("tab10")
    for index, cluster in enumerate([-1] + cluster_ids):
        mask = labels == cluster
        if not np.any(mask):
            continue
        color = "#555555" if cluster == -1 else palette(index % 10)
        name = "noise (-1)" if cluster == -1 else f"cluster {cluster}"
        ax.scatter(data.loc[mask, numeric_features[1]], data.loc[mask, numeric_features[0]],
                   s=18 if cluster == -1 else 22, alpha=0.55, color=color, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"DBSCAN — geometry clusters (ε={eps:.4f}, min_samples={min_samples})")
    ax.set_xlabel("Raw width (pixels, log scale)")
    ax.set_ylabel("Raw height (pixels, log scale)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.15, which="both")
    fig.tight_layout()
    cluster_plot_path = output_dir / "dbscan_geometry_clusters.png"
    fig.savefig(cluster_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    main_row = sensitivity.loc[
        (sensitivity["sensitivity_axis"] == "epsilon")
        & (sensitivity["min_samples"] == min_samples)
        & np.isclose(sensitivity["eps_multiplier"], 1.0)
    ].iloc[0]
    metrics = {
        "features_used": numeric_features,
        "target_used_to_fit": False,
        "rows_clustered": int(len(data)),
        "distance": "Euclidean after log1p and RobustScaler (IQR scaling) on median-imputed geometry",
        "distance_rationale": (
            "Height, width, and aspect ratio have different units and skewed ranges. "
            "log1p limits the influence of large values; RobustScaler centers by the "
            "median and scales each dimension by its IQR. Euclidean distance then "
            "treats a one-IQR change in each transformed geometry field comparably. "
            "This is a reproducible baseline, not a domain-calibrated visual similarity metric."
        ),
        "missing_value_handling": "median imputation fit on the full unlabeled geometry table for this descriptive unsupervised run",
        "epsilon_method": "maximum perpendicular distance from the sorted k-distance curve to its endpoint chord",
        "epsilon_rationale": (
            "The epsilon is selected from the observed 10th-nearest-neighbor distance "
            "curve using its maximum distance from the endpoint chord. This elbow rule "
            "is a data-driven heuristic; the saved sensitivity sweep varies epsilon "
            "by 0.75x, 1.0x, 1.25x, and 1.5x."
        ),
        "k_for_k_distance": min_samples,
        "k_distance_curve_knee_rank_zero_based": int(knee_index),
        "k_distance_curve_knee_fraction": float(knee_index / max(len(sorted_k_distances) - 1, 1)),
        "eps_selected": float(eps),
        "min_samples_selected": min_samples,
        "min_samples_rationale": (
            "There are 3 geometry dimensions; min_samples=10 is greater than d+1, "
            "so a core point needs a denser neighborhood than the minimum dimension-based "
            "rule of thumb. The actual run is compared with 6 and 15."
        ),
        "clusters_excluding_noise": int(len(cluster_ids)),
        "noise_count": noise_count,
        "noise_fraction": noise_count / len(labels),
        "silhouette_non_noise_sampled": main_row["silhouette_non_noise_sampled"],
        "adjusted_rand_vs_class_all_rows_posthoc": float(adjusted_rand_score(y_external, labels)),
        "adjusted_rand_vs_class_non_noise_posthoc": (
            float(adjusted_rand_score(y_external[labels != -1], labels[labels != -1]))
            if np.sum(labels != -1) > 0 else None
        ),
        "silhouette_method": "Euclidean silhouette on non-noise rows; if more than 1,200 non-noise rows, a fixed-seed sample of 1,200 is used.",
        "class_label_use": "external post-hoc profile only; class was excluded from imputation, scaling, distance, and DBSCAN fit",
        "eps_sensitivity_multipliers": list(EPS_MULTIPLIERS),
        "min_samples_sensitivity": list(DBSCAN_MIN_SAMPLES_SENSITIVITY),
        "sensitivity_results": sensitivity.to_dict(orient="records"),
        "figures": {"k_distance": str(kdist_path), "clusters": str(cluster_plot_path)},
        "noise_file": str(output_dir / "dbscan_noise_points.csv"),
        "sensitivity_file": str(output_dir / "dbscan_parameter_sensitivity.csv"),
    }
    (output_dir / "dbscan_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return {"metrics": metrics, "sensitivity": sensitivity,
            "cluster_profile": profile, "output_dir": str(output_dir)}


def run_analysis(
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = ARTIFACT_DIR,
) -> dict:
    """Run both requested analyses and persist actual metrics/artifacts."""
    data, feature_names, numeric_features, binary_features = load_ads(raw_dir)
    signatures = feature_signatures(data, feature_names)
    signature_counts = signatures.value_counts()
    signature_target_counts = data.groupby(signatures)["class"].nunique()
    tree_results = run_decision_tree(
        data, numeric_features, binary_features, feature_names, output_dir
    )
    dbscan_results = run_dbscan(data, numeric_features, output_dir)
    summary = {
        "dataset": {
            "rows": len(data),
            "columns_including_target": data.shape[1],
            "features": len(feature_names),
            "target_counts": data["class"].value_counts().to_dict(),
            "repeated_feature_signatures": {
                "distinct_groups": int(signature_counts.size),
                "groups_with_repeated_rows": int((signature_counts > 1).sum()),
                "rows_in_groups_with_repeats": int(signature_counts[signature_counts > 1].sum()),
                "extra_rows_beyond_one_per_signature": int((signature_counts - 1).clip(lower=0).sum()),
                "groups_with_conflicting_target_labels": int((signature_target_counts > 1).sum()),
                "handling": "All rows are retained. Group-aware supervised splitting keeps each feature signature in one partition.",
            },
            "raw_file_manifest": raw_file_manifest(raw_dir),
        },
        "environment": _package_versions(),
        "decision_tree": tree_results["metrics"],
        "decision_tree_rules": tree_results["rules"],
        "dbscan": dbscan_results["metrics"],
    }
    (output_dir / "classification_clustering_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return {"summary": summary, "tree": tree_results, "dbscan": dbscan_results}


def _main() -> None:
    results = run_analysis()
    print("Decision Tree metrics:")
    print(json.dumps(results["summary"]["decision_tree"], indent=2))
    print("Decision Tree IF-THEN rules:")
    for rule in results["tree"]["rules"]:
        print(rule["text"])
    print("DBSCAN metrics:")
    print(json.dumps(results["summary"]["dbscan"], indent=2))
    print("DBSCAN parameter sensitivity:")
    print(results["dbscan"]["sensitivity"].to_string(index=False))
    print("DBSCAN cluster/class profile:")
    print(results["dbscan"]["cluster_profile"].to_string())


if __name__ == "__main__":
    _main()
