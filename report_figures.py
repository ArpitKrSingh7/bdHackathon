"""Generate interpreted report figures from the checked-in raw data and outputs."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from scipy.cluster.hierarchy import dendrogram, linkage

from hackathon_utils import RAW_DATA_DIR, load_ads

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "artifacts" / "models"
ASSOCIATION_DIR = PROJECT_DIR / "artifacts" / "association"
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "report" / "figures"
COLORS = {"nonad": "#416F9B", "ad": "#CF6E53"}


def _save(fig, name: str, chart_type: str, caption: str, entries: list) -> None:
    path = OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    entries.append({
        "file": str(path.relative_to(PROJECT_DIR)),
        "chart_type": chart_type,
        "caption": caption,
    })


def generate_report_figures() -> list[dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, feature_names, geometry, binary = load_ads(RAW_DATA_DIR)
    entries: list[dict] = []
    counts = data["class"].value_counts().reindex(["nonad", "ad"])
    ratio = float(counts["nonad"] / counts["ad"])

    # 1. Bar chart of target counts.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(counts.index, counts.values, color=[COLORS[c] for c in counts.index])
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set(title="Internet Ads class counts", xlabel="Class", ylabel="Rows")
    ax.set_ylim(0, counts.max() * 1.13)
    _save(fig, "01_class_counts.png", "bar chart",
          f"The raw file contains {counts['nonad']:,} nonads and {counts['ad']:,} ads, a {ratio:.2f}:1 imbalance. Class-aware metrics are therefore needed alongside accuracy.", entries)

    # 2. Pie chart of class proportions.
    fig, ax = plt.subplots(figsize=(5.3, 4.2))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90,
           colors=[COLORS[c] for c in counts.index], wedgeprops={"edgecolor": "white"})
    ax.set_title("Share of rows by class")
    _save(fig, "02_class_share.png", "pie chart",
          f"Ads account for {counts['ad'] / len(data):.1%} of the 3,279 rows, so a majority-only classifier would hide poor ad detection.", entries)

    # 3. Lollipop chart of missing values.
    missing = data[feature_names].isna().sum().loc[lambda s: s.gt(0)].sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hlines(missing.index, 0, missing.values, color="#A99AC4", linewidth=2)
    ax.scatter(missing.values, missing.index, color="#62518A", s=45, zorder=3)
    ax.set_xscale("log")
    ax.set(title="Missing values in affected attributes (log count axis)", xlabel="Missing rows (log scale)", ylabel="Attribute")
    for name, value in missing.items():
        ax.text(value * 1.08, name, f"{value} ({value / len(data):.1%})", va="center", fontsize=8)
    ax.set_xlim(10, float(missing.max()) * 2.8)
    _save(fig, "03_missingness_lollipop.png", "lollipop chart",
          f"Geometry is partly unknown in {int(data[geometry].isna().any(axis=1).sum()):,} rows; the separate local indicator is missing in {int(data['local'].isna().sum())} rows. Missing values were kept distinct from observed zeroes.", entries)

    # 4. Histograms on log1p axes for the geometry fields.
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))
    geometry_skew = data[geometry].skew()
    for name, ax in zip(geometry, axes):
        values = data[name].dropna().astype(float)
        ax.hist(np.log1p(values), bins=28, color="#5485A9", edgecolor="white")
        ax.set(title=name, xlabel="log1p(value)", ylabel="Rows")
    fig.suptitle("Geometry distributions after log1p display transform", y=1.05)
    _save(fig, "04_geometry_histograms.png", "histogram",
          "The log1p axes make the long upper tails visible without removing large measurements. Raw skewness is " + ", ".join(f"{name} {geometry_skew[name]:.2f}" for name in geometry) + "; the transform is only for display, and raw values remain in the saved cleaned copy.", entries)

    # 5. Box plots by class.
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    medians = data.groupby("class")[geometry].median()
    for name, ax in zip(geometry, axes):
        groups = [data.loc[data["class"].eq(label), name].dropna() for label in ("nonad", "ad")]
        bp = ax.boxplot(groups, labels=["nonad", "ad"], showfliers=False, patch_artist=True)
        for patch, label in zip(bp["boxes"], ("nonad", "ad")):
            patch.set_facecolor(COLORS[label])
            patch.set_alpha(0.65)
        ax.set(title=name, xlabel="Class", ylabel="Raw value")
    fig.suptitle("Geometry by class (box plots; fliers omitted for scale)", y=1.05)
    median_text = ", ".join(f"{name}: ad {medians.loc['ad', name]:g}, nonad {medians.loc['nonad', name]:g}" for name in geometry)
    _save(fig, "05_geometry_boxplots.png", "box plot",
          f"Observed medians (ad / nonad) are {median_text}. These descriptive differences overlap and do not by themselves define a classifier.", entries)

    # 6. Scatter plot of width and height by label.
    points = data[["width", "height", "class"]].dropna()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for label in ("nonad", "ad"):
        group = points[points["class"].eq(label)]
        ax.scatter(group["width"], group["height"], s=13, alpha=0.30,
                   color=COLORS[label], label=f"{label} (n={len(group):,})", rasterized=True)
    ax.set(title="Image width vs height by class", xlabel="Width (pixels)", ylabel="Height (pixels)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    corr = float(points[["width", "height"]].corr(method="spearman").iloc[0, 1])
    _save(fig, "06_width_height_scatter.png", "scatter plot",
          f"For rows with both dimensions, Spearman width-height correlation is {corr:.3f}. The cloud shows substantial geometric spread; the plot is descriptive, not evidence of causation.", entries)

    # 7. Dot plot of prevalent binary indicators.
    prevalence = data[binary].eq(1).mean().sort_values(ascending=False).head(18).sort_values()
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    ax.scatter(prevalence.values, prevalence.index, s=42, color="#3D8A73")
    ax.set(title="Most prevalent binary indicators", xlabel="Fraction of all rows with value 1", ylabel="Feature", xlim=(0, max(prevalence) * 1.13))
    ax.grid(axis="x", alpha=0.22)
    top_three = data[binary].eq(1).mean().sort_values(ascending=False).head(3)
    top_three_text = ", ".join(f"{name} {value:.1%}" for name, value in top_three.items())
    _save(fig, "07_binary_prevalence_dot.png", "dot plot",
          f"The three most prevalent observed indicators are {top_three_text}. This descriptive ranking motivated a support-filtered transaction alphabet; it is not a model feature-importance ranking.", entries)

    # 8. Correlation heatmap among geometry fields.
    corr_matrix = data[geometry].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(5.7, 4.7))
    image = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(geometry)), geometry, rotation=25, ha="right")
    ax.set_yticks(range(len(geometry)), geometry)
    for i in range(len(geometry)):
        for j in range(len(geometry)):
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    ax.set_title("Spearman correlation: geometry fields")
    fig.colorbar(image, ax=ax, label="Rank correlation")
    pairs = [(geometry[i], geometry[j], corr_matrix.iloc[i, j]) for i in range(len(geometry)) for j in range(i + 1, len(geometry))]
    pair_text = "; ".join(f"{a}/{b} {value:.2f}" for a, b, value in pairs)
    _save(fig, "08_geometry_correlation_heatmap.png", "correlation heatmap",
          f"Spearman correlations are {pair_text}. Correlation describes monotonic association, not causation; missing pairs are omitted.", entries)

    # 9. Stem-and-leaf display using a reproducible 100-row sample.
    height_values = data["height"].dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(42)
    sampled = np.rint(rng.choice(height_values, size=min(100, len(height_values)), replace=False)).astype(int)
    leaves = {}
    for value in sampled:
        leaves.setdefault(int(value // 10), []).append(int(value % 10))
    stems = sorted(leaves)
    fig_height = max(4.2, 0.19 * len(stems) + 1.3)
    fig, ax = plt.subplots(figsize=(8.4, fig_height))
    ax.axis("off")
    ax.set_title("Stem-and-leaf display: sampled image heights", pad=14)
    for row, stem in enumerate(stems):
        ax.text(0.17, 0.96 - row * 0.88 / max(1, len(stems) - 1), f"{stem:>4} | {' '.join(map(str, sorted(leaves[stem])))}",
                family="monospace", fontsize=8.5, transform=ax.transAxes)
    ax.text(0.17, 0.03, "Key:  6 | 4 = 64 pixels; fixed-seed sample, n=" + str(len(sampled)),
            transform=ax.transAxes, fontsize=8.5)
    stem_frequency = pd.Series(sampled // 10).value_counts()
    busiest_stem = int(stem_frequency.idxmax())
    _save(fig, "09_height_stem_leaf.png", "stem-and-leaf display",
          f"In the fixed-seed sample of {len(sampled)}, heights range from {sampled.min()} to {sampled.max()} pixels; stem {busiest_stem} (the {busiest_stem * 10}s) occurs most often ({int(stem_frequency.max())} values). It is a visualization sample only; all source rows remain in the analyses.", entries)

    # 10. Violin plot for the strongly skewed width field on log1p scale.
    width_groups = [np.log1p(data.loc[data["class"].eq(label), "width"].dropna().to_numpy(dtype=float)) for label in ("nonad", "ad")]
    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    violins = ax.violinplot(width_groups, positions=[1, 2], showmeans=False, showmedians=True, showextrema=True)
    for body, label in zip(violins["bodies"], ("nonad", "ad")):
        body.set_facecolor(COLORS[label]); body.set_alpha(0.6)
    ax.set_xticks([1, 2], ["nonad", "ad"])
    ax.set(title="Width density by class", xlabel="Class", ylabel="log1p(width)")
    _save(fig, "10_width_violin.png", "violin plot",
          f"The violin shows the width distribution's shape on a log1p axis; raw median widths are {medians.loc['nonad', 'width']:g} pixels for nonads and {medians.loc['ad', 'width']:g} for ads. Broad spread means median separation is not perfect classification.", entries)

    # 11. ECDF by class.
    fig, ax = plt.subplots(figsize=(7.1, 4.6))
    for label in ("nonad", "ad"):
        vals = np.sort(data.loc[data["class"].eq(label), "aratio"].dropna().to_numpy(dtype=float))
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", color=COLORS[label], label=f"{label} (n={len(vals):,})")
    ax.set(title="Empirical CDF of aspect ratio by class", xlabel="Aspect ratio (raw units)", ylabel="Cumulative fraction", xlim=(0, None), ylim=(0, 1.02))
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    _save(fig, "11_aratio_ecdf.png", "empirical CDF",
          f"The CDF compares the full aspect-ratio distributions and makes class quantiles visible. The raw medians are ad {medians.loc['ad', 'aratio']:g} and nonad {medians.loc['nonad', 'aratio']:g}; missing values are omitted from this plot.", entries)

    # 12. Association rule support-confidence bubble plot.
    rules = pd.read_csv(ASSOCIATION_DIR / "apriori_class_rules.csv")
    if not rules.empty:
        plot_rules = rules
        fig, ax = plt.subplots(figsize=(7.3, 5.0))
        scatter = ax.scatter(plot_rules["support"], plot_rules["confidence"],
                              s=np.clip(8 + 0.06 * plot_rules["support_count"], 10, 28),
                              c=plot_rules["lift"], cmap="viridis", alpha=0.30, linewidths=0)
        ax.set(title="All Apriori class rules: support vs confidence", xlabel="Support", ylabel="Confidence", ylim=(0.55, 1.02))
        fig.colorbar(scatter, ax=ax, label="Lift")
        best = rules.sort_values(["lift", "confidence"], ascending=False).iloc[0]
        ax.annotate(f"highest-lift rule: lift={best['lift']:.2f}", (best["support"], best["confidence"]), xytext=(8, -18), textcoords="offset points", fontsize=8)
        _save(fig, "12_apriori_rules_bubble.png", "bubble plot",
              f"The plot shows all {len(plot_rules):,} class-consequent rules; marker size reflects support count and color is lift. Many rules share the same support/confidence values. The highest-lift rule has support {best['support']:.3f}, confidence {best['confidence']:.3f}, and lift {best['lift']:.3f}; association is not causation.", entries)

    # 13. Closed itemset size profile (horizontal lollipop).
    closed = pd.read_csv(ASSOCIATION_DIR / "charm_closed_itemsets.csv")
    lengths = closed["length"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.1, 4.2))
    ax.hlines(lengths.index, 0, lengths.values, color="#B197C9", linewidth=2)
    ax.scatter(lengths.values, lengths.index, color="#6E4D8E", s=38)
    ax.set(title="CHARM closed itemsets by cardinality", xlabel="Closed itemsets", ylabel="Number of items")
    _save(fig, "13_charm_closed_lengths.png", "lollipop chart",
          f"Feature-only CHARM found {len(closed):,} closed itemsets at 5% support. The most common closure size is {int(lengths.idxmax())} items ({int(lengths.max())} closures). Closed sets are a compact representation and the output was verified against exhaustive TID intersections.", entries)

    # 14. Confusion matrix from executed held-out predictions.
    prediction = pd.read_csv(MODEL_DIR / "decision_tree_test_predictions.csv")
    labels = ["nonad", "ad"]
    cm = np.array([[((prediction.actual == truth) & (prediction.predicted == pred)).sum() for pred in labels] for truth in labels])
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels); ax.set_yticks([0, 1], labels)
    ax.set(xlabel="Predicted", ylabel="Actual", title="Decision Tree held-out confusion matrix")
    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if cm[i, j] > threshold else "black", fontsize=12)
    fig.colorbar(image, ax=ax, label="Rows")
    _save(fig, "14_tree_confusion_matrix.png", "confusion matrix",
          f"On the grouped test set, the tree identifies {cm[1, 1]} of {cm[1].sum()} ads and misses {cm[1, 0]}; it flags {cm[0, 1]} nonads as ads. This threshold-dependent view complements ROC-AUC.", entries)

    # 15. ROC curve reconstructed from actual held-out probabilities.
    y_true = prediction["actual"].eq("ad").astype(int).to_numpy()
    y_score = prediction["ad_probability"].to_numpy(dtype=float)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.plot(fpr, tpr, color="#4C8A78", linewidth=2, label=f"Decision Tree (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Chance")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate", title="Receiver operating characteristic")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.2)
    _save(fig, "15_tree_roc_curve.png", "ROC curve",
          f"The held-out ROC-AUC is {auc:.4f}. It measures ranking over thresholds and should be read with ad precision/recall because the classes are imbalanced.", entries)

    # 16–18. Existing figures generated by the actual DBSCAN/tree runs.
    copied = [
        ("decision_tree_structure.png", "16_decision_tree_diagram.png", "decision tree diagram",
         "The fitted depth-5 CART structure gives 12 terminal IF-THEN rules. Width, height, and URL/anchor indicators are split variables; the rule list is also provided as a separate CSV and text file."),
        ("dbscan_k_distance.png", "17_dbscan_k_distance.png", "k-distance curve",
         "The 10-neighbor distance elbow selected epsilon=0.4578. Epsilon is a heuristic from the unlabeled distance curve; the saved sensitivity sweep shows how nearby settings change noise and clusters."),
        ("dbscan_geometry_clusters.png", "18_dbscan_clusters.png", "cluster projection scatter plot",
         "The feature-only DBSCAN run found 3 clusters and 70 noise rows (2.13%). The target was excluded from distance construction and clustering; class ARI is reported only as post-hoc context."),
    ]
    for source, name, chart_type, caption in copied:
        shutil.copyfile(MODEL_DIR / source, OUTPUT_DIR / name)
        entries.append({"file": str((OUTPUT_DIR / name).relative_to(PROJECT_DIR)), "chart_type": chart_type, "caption": caption})

    # 19. Measured local/Spark task-fit timings (actual execution only).
    spark_summary_path = PROJECT_DIR / "artifacts" / "spark" / "pyspark_benchmark_summary.json"
    spark_summary = json.loads(spark_summary_path.read_text(encoding="utf-8"))
    if spark_summary.get("status") == "completed":
        local_tree = spark_summary["local_tree"]
        local_arm = spark_summary["local_apriori"]
        spark_tree = spark_summary["spark_tree"]
        spark_arm = spark_summary["spark_fpgrowth"]
        local_times = [local_tree["fit_seconds_median"], local_arm["elapsed_seconds"]]
        spark_times = [spark_tree["fit_seconds_median"], spark_arm["fit_seconds_median"]]
        x = np.arange(2)
        width = 0.34
        fig, ax = plt.subplots(figsize=(7.5, 4.7))
        local_bars = ax.bar(x - width / 2, local_times, width, label="Local", color="#3F779E")
        spark_bars = ax.bar(x + width / 2, spark_times, width, label="PySpark", color="#D17B59")
        ax.set_yscale("log")
        ax.set_xticks(x, ["Decision Tree fit", "Apriori / FP-Growth fit"])
        ax.set_ylabel("Median task time (seconds, log scale)")
        ax.set_title("Measured local and PySpark task-fit times")
        ax.legend(frameon=False)
        ax.grid(axis="y", which="both", alpha=0.2)
        for bars in (local_bars, spark_bars):
            ax.bar_label(bars, labels=[f"{bar.get_height():.4f}s" for bar in bars], padding=3, fontsize=8)
        tree_ratio = spark_times[0] / local_times[0]
        arm_ratio = spark_times[1] / local_times[1]
        tree_e2e = spark_summary["tree_timing_comparison"]["spark_end_to_end_excluding_startup_seconds"]
        arm_e2e = spark_summary["arm_timing_comparison"]["spark_fp_end_to_end_excluding_startup_seconds"]
        startup = spark_summary["spark_startup_seconds"]
        _save(fig, "19_spark_runtime_comparison.png", "log-scale grouped bar chart",
              f"Spark median fit was {tree_ratio:.1f}× local for the Decision Tree ({spark_times[0]:.4f} vs {local_times[0]:.4f} s) and {arm_ratio:.1f}× for FP-Growth fit vs Apriori mining ({spark_times[1]:.4f} vs {local_times[1]:.4f} s; algorithms differ). Including Spark conversion/materialization and prediction/collection, end-to-end task times were {tree_e2e:.3f} s for the tree and {arm_e2e:.3f} s for pattern mining, excluding the shared {startup:.3f} s startup. Spark was slower on this dataset.", entries)

    # 20. Course dendrogram as a separate geometry-only visualization, not DBSCAN output.
    geometry_matrix = data[geometry].to_numpy(dtype=float)
    geometry_matrix = SimpleImputer(strategy="median").fit_transform(geometry_matrix)
    geometry_matrix = np.log1p(geometry_matrix)
    geometry_matrix = RobustScaler().fit_transform(geometry_matrix)
    rng = np.random.default_rng(42)
    selected = np.sort(rng.choice(len(geometry_matrix), size=min(120, len(geometry_matrix)), replace=False))
    linkage_matrix = linkage(geometry_matrix[selected], method="average", metric="euclidean")
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    dendrogram(linkage_matrix, truncate_mode="lastp", p=30, show_contracted=True,
               no_labels=True, color_threshold=None, ax=ax)
    ax.set(title="Average-linkage dendrogram of a geometry-only sample",
           xlabel="Sampled row groups (truncated to final 30 branches)",
           ylabel="Euclidean distance after median imputation, log1p, and IQR scaling")
    final_merge = float(linkage_matrix[-1, 2])
    _save(fig, "20_geometry_dendrogram_sample.png", "dendrogram (supplementary course visualization)",
          f"For a fixed-seed sample of {len(selected)} rows, average linkage's final merge occurs at transformed distance {final_merge:.3f}. This course visualization uses geometry only; it is separate from DBSCAN and does not use class labels or define the reported DBSCAN clusters.", entries)

    manifest_path = OUTPUT_DIR.parent / "figure_manifest.json"
    manifest_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return entries


if __name__ == "__main__":
    figures = generate_report_figures()
    print(json.dumps({"figure_count": len(figures), "chart_types": list(dict.fromkeys(x["chart_type"] for x in figures)), "manifest": str(OUTPUT_DIR.parent / "figure_manifest.json")}, indent=2))
