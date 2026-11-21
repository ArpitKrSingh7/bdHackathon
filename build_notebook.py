"""Integrate analysis outputs and execute the notebook in one Python process.

The workspace denies local sockets needed to start an ordinary Jupyter kernel.
This builder executes each notebook code cell sequentially in one Python
namespace and saves the real stdout, displayed tables, and figures as outputs.
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import nbformat

PROJECT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_DIR / "internet_ads_hackathon.ipynb"


def _cell_source(cell, source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def _markdown(source: str):
    return nbformat.v4.new_markdown_cell(source)


def _code(source: str):
    return nbformat.v4.new_code_cell(source)


def _set_existing_markdown(nb, index: int, source: str) -> None:
    _cell_source(nb.cells[index], source)


def _append_integrated_sections(nb) -> None:
    nb.cells.extend([
        _markdown("""## 6. Saved preprocessing and repeated signatures

The cleaned copy below normalizes feature types and labels but retains every row and every missing value. Task-specific imputation is fitted only where the analysis requires it: on training rows for the Decision Tree, and on the unlabeled geometry table for descriptive DBSCAN. `preprocessing_config.json` records those choices and the raw-file SHA-256 manifest."""),
        _code("""from pathlib import Path
import json
import pandas as pd

PREPROCESSED_DIR = WORK_DIR / "artifacts" / "processed"
preprocessing_config = json.loads((PREPROCESSED_DIR / "preprocessing_config.json").read_text())
cleaned_saved = pd.read_csv(PREPROCESSED_DIR / "internet_ads_cleaned.csv")
signature_ledger = pd.read_csv(PREPROCESSED_DIR / "feature_signature_ledger.csv")
repeats = preprocessing_config["repeated_feature_signatures"]
print(f"Saved cleaned copy: {cleaned_saved.shape[0]:,} rows × {cleaned_saved.shape[1]:,} columns")
print(f"Missing feature cells preserved: {int(cleaned_saved[feature_names].isna().sum().sum()):,}")
print(f"Distinct full-feature signatures: {repeats['distinct_signatures']:,}")
print(f"Repeated signature groups: {repeats['groups_with_repeated_rows']:,}; rows in those groups: {repeats['rows_in_repeated_groups']:,}; excess rows: {repeats['extra_rows_beyond_first_per_signature']:,}")
print(f"Conflicting-label signature groups: {repeats['groups_with_conflicting_labels']}")
print("No repeated row was dropped. The row-level ledger is saved beside the cleaned copy.")
display(cleaned_saved[[*NUMERIC_FEATURES, "local", TARGET]].head())"""),
        _markdown("""## 7. Apriori and CHARM: feature mining separated from class rules

Each raw row is a transaction, including repeated signatures. The three continuous geometry columns are excluded rather than binned. The primary Apriori and CHARM runs use feature-only transactions. Separate target-augmented runs produce class-consequent rules; this keeps `class` out of the unsupervised itemsets. At 5% support the minimum count is 164 rows, and rule confidence is at least 60%."""),
        _code("""ASSOCIATION_DIR = WORK_DIR / "artifacts" / "association"
association_metadata = json.loads((ASSOCIATION_DIR / "association_metadata.json").read_text())
feature_mining = association_metadata["feature_only_mining"]
class_mining = association_metadata["class_association_rule_mining"]
feature_itemsets = pd.read_csv(ASSOCIATION_DIR / "apriori_itemsets.csv")
feature_rules = pd.read_csv(ASSOCIATION_DIR / "apriori_rules.csv")
class_rules = pd.read_csv(ASSOCIATION_DIR / "apriori_class_rules.csv")
closed_itemsets = pd.read_csv(ASSOCIATION_DIR / "charm_closed_itemsets.csv")
charm_class_rules = pd.read_csv(ASSOCIATION_DIR / "charm_class_rules.csv")
print(f"Transactions: {association_metadata['rows_as_transactions']:,}; selected feature items: {association_metadata['selected_binary_feature_count']}")
print(f"Feature-only Apriori: {len(feature_itemsets):,} itemsets, {len(feature_rules):,} rules; runtime {feature_mining['apriori_runtime_seconds_excluding_transaction_build']:.6f} s")
print(f"Feature-only CHARM: {len(closed_itemsets):,} closed itemsets; exhaustive verification={feature_mining['charm_exhaustive_verification']['matches_exhaustive']}")
print(f"Separate target-augmented Apriori: {len(class_rules):,} class-consequent rules from {class_mining['apriori_itemset_count']:,} itemsets")
print(f"Separate target-augmented CHARM: {len(charm_class_rules):,} class rules from {class_mining['charm_class_augmented_closed_itemset_count']:,} closures; exhaustive verification={class_mining['charm_exhaustive_verification']['matches_exhaustive']}")
print("Top class-consequent Apriori rules, ranked by lift:")
display(class_rules[["antecedent_text", "consequent_text", "support_count", "support", "confidence", "lift"]].head(8))
print("Feature-only Apriori rules:")
display(feature_rules[["antecedent_text", "consequent_text", "support_count", "support", "confidence", "lift"]].head(5))"""),
        _code("""support_sensitivity = pd.read_csv(ASSOCIATION_DIR / "apriori_support_sensitivity.csv")
print("Sensitivity runs cap the union itemset at length 3; the primary runs are unlimited length.")
display(support_sensitivity)"""),
        _markdown("""### Sequential-pattern requirement: structural analysis and separate synthetic demo

The Internet Ads file has no timestamp, session identifier, or natural event order, so it cannot support temporal or session-based sequence claims. A separate data-derived analysis orders active indicators by the URL-named field families `origurl → ancurl → url`; this is an analyst-defined structural order, not observed chronology, and may not satisfy a rubric requiring temporal sequences. The five-sequence GSP example below is synthetic and remains separate from the dataset."""),
        _code("""toy_sequence = association_metadata["toy_sequential_demo"]
toy_patterns = pd.read_csv(ASSOCIATION_DIR / "toy_sequential_patterns.csv")
print(f"Synthetic sequences: {toy_sequence['sequence_count']}; minimum support={toy_sequence['parameters']['min_support']:.0%} ({toy_sequence['parameters']['min_support_count']} sequences); patterns={len(toy_patterns)}")
print("Hand-authored toy sequences:")
for i, sequence in enumerate(toy_sequence["sequences"], start=1):
    print(f"  {i}: {' → '.join(sequence)}")
display(toy_patterns.head(12))"""),
        _markdown("""## 8. Decision Tree: grouped holdout, train-only preprocessing, and rules

All copies of the same full input signature stay in one partition. Median imputation for geometry and mode imputation for binary fields are fitted on training rows only; the test set is not used to fit preprocessing. Balanced class weights address the ad/nonad imbalance. Rule probabilities below are the predicted class share of class-weighted training mass at a leaf, not the unweighted fraction of ads."""),
        _code("""MODEL_DIR = WORK_DIR / "artifacts" / "models"
model_summary = json.loads((MODEL_DIR / "classification_clustering_summary.json").read_text())
tree_result = model_summary["decision_tree"]
print(f"Split: {tree_result['n_train']:,} train / {tree_result['n_test']:,} test; feature-signature overlap={tree_result['split']['signature_overlap']}")
print("Preprocessing:", tree_result["preprocessing"])
print("Parameters:", tree_result["tree_parameters"])
for metric in ["accuracy", "balanced_accuracy", "precision_ad", "recall_ad", "f1_ad", "roc_auc"]:
    print(f"{metric}: {tree_result[metric]:.4f}")
print("Confusion matrix (actual nonad/ad by predicted nonad/ad):", tree_result["confusion_matrix_labels_nonad_ad"])
tree_rules = pd.read_csv(MODEL_DIR / "decision_tree_if_then_rules.csv")
print(f"Extracted IF-THEN leaf rules: {len(tree_rules)}")
display(tree_rules[["rule_id", "if", "then", "training_rows_at_leaf", "weighted_predicted_class_probability_at_leaf"]])"""),
        _markdown("""## 9. DBSCAN: geometry distance, epsilon, noise, and sensitivity

DBSCAN uses height, width, and aspect ratio only. After median imputation, `log1p` and RobustScaler place the three differently scaled/skewed fields on comparable IQR scales; Euclidean distance is then used. A 10-neighbor distance-curve knee chose epsilon. Class is excluded from fitting and appears only in post-hoc profile/ARI results. The parameter sweep is included because DBSCAN assignments are sensitive to epsilon and `min_samples`."""),
        _code("""dbscan_result = model_summary["dbscan"]
print("Distance:", dbscan_result["distance"])
print(f"epsilon={dbscan_result['eps_selected']:.6f}; min_samples={dbscan_result['min_samples_selected']}; clusters={dbscan_result['clusters_excluding_noise']}")
print(f"Noise: {dbscan_result['noise_count']} / {dbscan_result['rows_clustered']} ({dbscan_result['noise_fraction']:.2%})")
print(f"Sampled non-noise silhouette={dbscan_result['silhouette_non_noise_sampled']:.4f}; post-hoc ARI vs class={dbscan_result['adjusted_rand_vs_class_all_rows_posthoc']:.4f}")
dbscan_sensitivity = pd.read_csv(MODEL_DIR / "dbscan_parameter_sensitivity.csv")
cluster_profile = pd.read_csv(MODEL_DIR / "dbscan_cluster_class_profile.csv")
noise_points = pd.read_csv(MODEL_DIR / "dbscan_noise_points.csv")
print("Parameter sensitivity:")
display(dbscan_sensitivity)
print("Cluster sizes and post-hoc class profile:")
display(cluster_profile)
print(f"Saved noise-point rows: {len(noise_points)}")
display(noise_points.head(10))"""),
        _markdown("""## 10. PySpark implementations and measured execution status

The source includes PySpark `DecisionTreeClassifier` on the same grouped split and train-only transformed matrices as scikit-learn, plus Spark FP-Growth on the same class-augmented row transactions and thresholds as local Apriori. Both PySpark workloads executed in local mode. On this small dataset, Spark was slower than the matched local baselines; startup and conversion costs are reported separately."""),
        _code("""SPARK_DIR = WORK_DIR / "artifacts" / "spark"
spark_summary = json.loads((SPARK_DIR / "pyspark_benchmark_summary.json").read_text())
local_tree = spark_summary["local_tree"]
local_apriori = spark_summary["local_apriori"]
print(f"Spark status: {spark_summary['status']}; PySpark {spark_summary['pyspark_version']}; master={spark_summary['spark_master']}")
startup_s = spark_summary.get("spark_startup_seconds", spark_summary.get("spark_startup_attempt_seconds"))
tasks_completed = spark_summary.get("spark_tasks_completed", spark_summary["status"] == "completed")
print(f"Spark startup: {startup_s:.3f} s; both workloads completed={tasks_completed}")
if spark_summary["status"] == "completed":
    spark_tree = spark_summary["spark_tree"]
    spark_fp = spark_summary["spark_fpgrowth"]
    tree_e2e = spark_summary["tree_timing_comparison"]["spark_end_to_end_excluding_startup_seconds"]
    fp_e2e = spark_summary["arm_timing_comparison"]["spark_fp_end_to_end_excluding_startup_seconds"]
    local_tree_e2e = local_tree["fit_seconds_median"] + local_tree["predict_seconds_median"]
    local_fp = local_apriori["elapsed_seconds"]
    comparison = pd.DataFrame([
        {"Task": "Decision Tree fit", "Local (s)": local_tree["fit_seconds_median"], "Spark (s)": spark_tree["fit_seconds_median"], "Spark/local": spark_tree["fit_seconds_median"] / local_tree["fit_seconds_median"]},
        {"Task": "Tree fit + predict/materialization", "Local fit+predict (s)": local_tree_e2e, "Spark end-to-end excluding startup (s)": tree_e2e, "Spark/local": tree_e2e / local_tree_e2e},
        {"Task": "Apriori vs FP-Growth fit", "Local Apriori (s)": local_fp, "Spark FP-Growth fit (s)": spark_fp["fit_seconds_median"], "Spark/local": spark_fp["fit_seconds_median"] / local_fp},
        {"Task": "Apriori vs FP-Growth end-to-end", "Local Apriori (s)": local_fp, "Spark FP-Growth including conversion/collection (s)": fp_e2e, "Spark/local": fp_e2e / local_fp},
    ])
    display(comparison)
    print(f"Tree Spark conversion/materialization: {spark_tree['spark_dataframe_conversion_and_materialization_seconds']:.3f} s; FP-Growth conversion/materialization: {spark_fp['spark_dataframe_conversion_and_materialization_seconds']:.3f} s; itemset/rule collection: {spark_fp['collect_itemsets_and_rules_seconds_median']:.3f} s.")
    print("Observed result: Spark was slower on this dataset. Spark startup is reported separately. FP-Growth and Apriori are different algorithms, so the association comparison is task-level.")
    print("Spark Decision Tree metrics:", spark_tree["metrics"])
    print("Matched local Decision Tree metrics:", local_tree["metrics"])
    print(f"Association outputs: local Apriori={local_apriori['frequent_itemset_count']:,} itemsets / {local_apriori['association_rule_count']:,} all-consequent rules; Spark FP-Growth={spark_fp['frequent_itemset_count']:,} itemsets / {spark_fp['association_rule_count']:,} singleton-consequent rules.")
    print("The 11,709 singleton-consequent local rules match Spark's rule set; the larger local total includes multi-item consequents.")
else:
    print(f"Spark tasks did not run: {spark_summary.get('spark_error', 'gateway initialization failed')}.")
    print("No Spark workload metric or local-vs-Spark performance conclusion is available.")"""),
        _markdown("""### 10.1 Feature selection and cleaning sensitivity

The ANOVA filter (`SelectKBest`, `k=100`) was fit on the training partition only and compared on the same grouped holdout as the all-feature tree. This is one holdout comparison, not cross-validated feature-selection evidence. The conflicting-label sensitivity copy excludes 54 rows from three exact feature-signature groups, but that screening uses target labels across the full file; it is not the primary analysis. IQR flags are review indicators, and all 363 flagged rows remain in the primary data."""),
        _code("""SENSITIVITY_DIR = WORK_DIR / "artifacts" / "models" / "preprocessing_sensitivity"
sensitivity = json.loads((SENSITIVITY_DIR / "preprocessing_sensitivity.json").read_text())
selection = sensitivity["classification_feature_selection"]
metric_names = ["accuracy", "balanced_accuracy", "precision_ad", "recall_ad", "f1_ad", "roc_auc"]
selection_rows = []
for label, result in [("All input features", selection["baseline_all_features"]), ("Top 100 ANOVA features", selection["selected_top_100"])]:
    row = {"Model": label, "Features": result["selected_feature_count"]}
    row.update({metric: result["metrics"][metric] for metric in metric_names})
    selection_rows.append(row)
display(pd.DataFrame(selection_rows).set_index("Model"))
conflicts = sensitivity["conflicting_label_cleaning_sensitivity"]
iqr_review = sensitivity["geometry_iqr_review"]
print(f"Conflict sensitivity only: {conflicts['conflict_groups']} groups / {conflicts['rows_excluded']} rows excluded; primary data unchanged.")
print(f"IQR geometry review: {iqr_review['rows_flagged_any']} rows flagged and {iqr_review['rows_deleted']} deleted.")
from IPython.display import Image, display
display(Image(filename=str(SENSITIVITY_DIR / "anova_feature_scores_top20.png")))"""),
        _markdown("""### 10.2 Dataset-derived URL-family pattern analysis

Each of the 3,279 actual rows is represented as three ordered sets of active binary indicators from `origurl`, `ancurl`, and `url`. The 16 selected items produced 71 frequent structural patterns and 29 prefix-extension rules at 5% support and 60% confidence (maximum length 3). The order is an explicit schema assumption; the source has neither raw URLs nor chronology, so these are not temporal findings."""),
        _code("""SEQUENCE_DIR = WORK_DIR / "artifacts" / "association"
sequence_metadata = json.loads((SEQUENCE_DIR / "schema_role_sequence_metadata.json").read_text())
sequence_patterns = pd.read_csv(SEQUENCE_DIR / "schema_role_sequential_patterns.csv")
sequence_rules = pd.read_csv(SEQUENCE_DIR / "schema_role_sequential_rules.csv")
print(f"Rows represented: {sequence_metadata['sequence_count']:,}; selected items: {sequence_metadata['selected_item_count']}; frequent patterns: {sequence_metadata['frequent_pattern_count']}; prefix-extension rules: {sequence_metadata['prefix_extension_rule_count']}.")
print("Order:", " → ".join(sequence_metadata["source_role_order"]))
print("Temporal sequence requirement fulfilled:", sequence_metadata["temporal_sequence_requirement_fulfilled"])
display(sequence_rules.sort_values(["confidence", "lift"], ascending=False).head(10))
display(Image(filename=str(SEQUENCE_DIR / "schema_role_patterns_top20.png")))"""),
        _markdown("""### 10.3 Repeated group-aware cross-validation

To check the single holdout result, the Decision Tree was evaluated with 5-fold `GroupKFold` over three repeats. Identical input signatures stay in one fold, and each fold fits imputation on its training partition only. These results supplement rather than replace the original holdout."""),
        _code("""CV_DIR = WORK_DIR / "artifacts" / "models" / "grouped_cv"
cv_summary = json.loads((CV_DIR / "grouped_cv_metrics.json").read_text())
cv_rows = []
for metric in ["accuracy", "balanced_accuracy", "precision_ad", "recall_ad", "f1_ad", "roc_auc"]:
    result = cv_summary["aggregated_metrics"][metric]
    cv_rows.append({"Metric": metric, "Mean": result["mean"], "SD": result["std"], "Min": result["min"], "Max": result["max"]})
print(f"{cv_summary['method']}; {cv_summary['total_folds_evaluated']} folds; {cv_summary['dataset_rows']:,} rows; {cv_summary['unique_signature_groups']:,} unique signature groups.")
print("Leakage checks:", cv_summary["group_leakage_check"], ";", cv_summary["imputation_leakage_check"])
display(pd.DataFrame(cv_rows).set_index("Metric"))"""),
        _markdown("""### 10.4 Synthetic PySpark scale sweep

This single-machine stress test repeats rows from the existing training partition; it does not add independent examples or new Internet Ads observations. One timing was recorded per size, so this is exploratory and not a stable scaling law. The measured scope is `local[2]`, not a multi-node cluster, and Spark remained slower at every successful size. A separate 1M-row attempt failed before model fitting and has no archived timing; 2M and 4M were not tested."""),
        _code("""SCALE_DIR = WORK_DIR / "artifacts" / "spark"
scale_metadata = json.loads((SCALE_DIR / "scaled_tree_benchmark.json").read_text())
scale = pd.read_csv(SCALE_DIR / "scaled_tree_benchmark.csv")
scale_view = pd.DataFrame({
    "Rows": scale["synthetic_training_rows"],
    "Local fit (s)": scale["local_fit_seconds_median"],
    "Spark fit (s)": scale["fit_seconds_median"],
    "Spark/local fit": scale["fit_seconds_median"] / scale["local_fit_seconds_median"],
    "Local expand + fit (s)": scale["local_expand_plus_fit_seconds"],
    "Spark end-to-end (s)": scale["spark_expand_convert_materialize_fit_seconds_excluding_startup"],
    "Spark/local end-to-end": scale["spark_expand_convert_materialize_fit_seconds_excluding_startup"] / scale["local_expand_plus_fit_seconds"],
})
print(f"Status: {scale_metadata['status']}; master={scale_metadata['spark_master']}; one observation per size={scale_metadata['repeats_per_size'] == 1}; startup={scale_metadata['spark_startup_seconds_separate']:.3f} s (excluded from per-size timings).")
display(scale_view.style.format({"Local fit (s)": "{:.4f}", "Spark fit (s)": "{:.4f}", "Spark/local fit": "{:.1f}×", "Local expand + fit (s)": "{:.4f}", "Spark end-to-end (s)": "{:.4f}", "Spark/local end-to-end": "{:.1f}×"}))
display(Image(filename=str(SCALE_DIR / "scaled_tree_benchmark.png")))"""),
        _markdown("""## 11. MLflow tracking and reproducibility

MLflow records the run parameters, measured metrics, and output files in a local SQLite database and a local artifact directory; no tracking server is required. It is useful for keeping experiments together during this short hackathon, but it does not replace this notebook, the PDF, or the source code. The tracked runs are refreshed from the final corrected artifacts before packaging."""),
                _code("""import mlflow
tracking_db = WORK_DIR / "artifacts" / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{tracking_db.resolve().as_posix()}")
experiment = mlflow.get_experiment_by_name("Internet Ads Hackathon (final artifacts)")
if experiment is None:
    print("No MLflow experiment found yet. Run `python3 log_experiments.py` after the final analysis outputs are generated.")
else:
    mlflow_runs = mlflow.search_runs([experiment.experiment_id], output_format="pandas")
    mlflow_runs = mlflow_runs.sort_values("start_time", ascending=False).head(4)
    cols = [c for c in ["tags.mlflow.runName", "run_id", "status", "metrics.accuracy", "metrics.ad_f1", "metrics.local_apriori_seconds", "metrics.spark_tree_fit_seconds_median", "metrics.spark_startup_seconds", "metrics.spark_tasks_completed"] if c in mlflow_runs.columns]
    print(f"Experiment: {experiment.name}; ID={experiment.experiment_id}; local database={tracking_db}")
    display(mlflow_runs[cols])"""),
        _markdown("""## 12. Numbered visualization index and individual inferences

The assignment handout asks for at least ten distinct plots with an inference for each. The 20 numbered report figures include the seven Lecture 4 forms (bar, pie, histogram, dot, stem-and-leaf, scatter, box), a Decision Tree diagram and ROC curve from classification material, and a supplementary geometry-only dendrogram from the clustering slides. A local-versus-Spark runtime chart and the other remaining figures are separately labeled diagnostics. Every caption below is computed from the source or saved analysis output.

The dendrogram is a fixed-seed, 120-row geometry-only illustration of a course plot. It is separate from DBSCAN and does not define the reported DBSCAN clusters."""),
    ])

    manifest = json.loads((PROJECT_DIR / "artifacts" / "report" / "figure_manifest.json").read_text())
    for number, item in enumerate(manifest, start=1):
        label = item["chart_type"].replace(" (supplementary course visualization)", "")
        nb.cells.append(_markdown(f"### Figure {number} — {label}\n\n{item['caption']}"))
        nb.cells.append(_code(f"""from IPython.display import Image, display
display(Image(filename=str(WORK_DIR / {item['file']!r})))"""))

    nb.cells.extend([
        _markdown("""## 13. Findings and limitations

- The source is imbalanced (about 86% nonad, 14% ad). On the original grouped holdout, the all-feature tree achieved ad recall 0.837 and ad F1 0.744. A separate training-only ANOVA top-100 sensitivity model reached ad F1 0.800 and precision 0.776 on the same holdout; feature selection was not cross-validated.
- Feature-only Apriori and CHARM reveal repeated URL/anchor-token combinations; class rules are generated separately and describe association, not causation.
- DBSCAN's selected unlabeled geometry setting found three clusters and marked 70 rows as noise. Its class alignment changes sharply with epsilon, so cluster/class agreement is not a stable finding.
- A data-derived `origurl → ancurl → url` pattern analysis used all rows, but its order is structural and analyst-defined, not temporal. The dataset still lacks timestamps, sessions, and event order. The synthetic GSP example is also only a demo; neither should be presented as observed user navigation.
- Fifteen-fold grouped CV supplements the holdout results. Both actual-data PySpark workloads and the synthetic repeated-row scale sweep are reported; Spark remained slower in the measured single-machine runs. The sweep has one timing per size and no model-quality estimate.

The project contains reusable scripts and generated artifacts. The LaTeX handoff is in `report_latex/`; the older ReportLab PDF and source ZIP predate the latest analyses and must not be treated as final."""),
        _markdown("""## 14. Re-run instructions and sources

Run the relevant analysis commands from this directory (`bdHackathon/`) to regenerate the current artifacts and notebook:

```bash
python3 scripts/data_quality_audit.py
python3 scripts/preprocessing_model_sensitivity.py
python3 scripts/schema_role_sequential_mining.py
python3 scripts/grouped_cv_decision_tree.py
python3 build_notebook.py
```

The notebook reads the saved PySpark scale-sweep artifact rather than rerunning the resource-intensive Spark job on every notebook build. Its reproducible implementation is `scripts/scaled_tree_spark_benchmark.py`; the saved sweep completed in `local[2]` through 500,000 synthetic rows. Re-running Spark requires a functioning local Java gateway. The final report is being authored separately in `report_latex/`; the older ReportLab PDF and ZIP are stale and are not rebuilt here.

**Project source:** the assignment handout and course PDFs supplied the rubric and course forms. The dataset is [UCI Internet Advertisements](https://archive.ics.uci.edu/dataset/51/internet%2Badvertisements) ([download archive](https://archive.ics.uci.edu/static/public/51/internet+advertisements.zip)). Algorithm reference: [CHARM, Zaki & Hsiao (2002)](https://doi.org/10.1137/1.9781611972726.27). Tracking reference: [MLflow Tracking](https://mlflow.org/docs/latest/tracking/)."""),
    ])


def _make_output(stdout: str, stderr: str, displays: list[dict]) -> list[dict]:
    outputs = []
    if stdout:
        outputs.append({"output_type": "stream", "name": "stdout", "text": stdout.splitlines(keepends=True)})
    if stderr:
        outputs.append({"output_type": "stream", "name": "stderr", "text": stderr.splitlines(keepends=True)})
    outputs.extend(displays)
    return outputs


def execute_in_process(nb) -> None:
    """Execute code cells sequentially and preserve actual text/table/plot outputs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython import display as ipy_display
    from IPython.display import Image as IPythonImage

    namespace = {"__name__": "__main__"}
    namespace["__file__"] = str(NOTEBOOK_PATH)
    display_buffer: list[dict] = []

    def capture_display(*objects, **kwargs):
        for obj in objects:
            if isinstance(obj, IPythonImage):
                raw = obj.data
                if isinstance(raw, str):
                    encoded = raw
                else:
                    encoded = base64.b64encode(raw).decode("ascii")
                display_buffer.append({"output_type": "display_data", "data": {"image/png": encoded}, "metadata": {}})
            elif isinstance(obj, pd.DataFrame):
                display_buffer.append({"output_type": "display_data", "data": {"text/html": obj.to_html(max_rows=100, max_cols=30), "text/plain": obj.to_string(max_rows=100, max_cols=30)}, "metadata": {}})
            elif isinstance(obj, pd.Series):
                display_buffer.append({"output_type": "display_data", "data": {"text/plain": obj.to_string()}, "metadata": {}})
            else:
                display_buffer.append({"output_type": "display_data", "data": {"text/plain": repr(obj)}, "metadata": {}})

    def capture_figures():
        for figure_number in list(plt.get_fignums()):
            fig = plt.figure(figure_number)
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight", facecolor="white")
            display_buffer.append({"output_type": "display_data", "data": {"image/png": base64.b64encode(buffer.getvalue()).decode("ascii")}, "metadata": {}})
            plt.close(fig)

    ipy_display.display = capture_display
    namespace["display"] = capture_display
    original_show = plt.show
    plt.show = lambda *args, **kwargs: capture_figures()

    count = 0
    try:
        for cell_number, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            count += 1
            cell["execution_count"] = count
            cell["outputs"] = []
            display_buffer.clear()
            stdout_buffer, stderr_buffer = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                    exec(compile(cell.source, f"<notebook-cell-{cell_number}>", "exec"), namespace)
                    capture_figures()
            except Exception:
                raise RuntimeError(f"Notebook code cell {cell_number} failed:\n{stdout_buffer.getvalue()}\n{stderr_buffer.getvalue()}") from sys.exc_info()[1]
            cell["outputs"] = [
                nbformat.from_dict(output)
                for output in _make_output(stdout_buffer.getvalue(), stderr_buffer.getvalue(), list(display_buffer))
            ]
    finally:
        plt.show = original_show


def build_notebook() -> dict:
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    # Remove the generated tail before rebuilding so reruns do not duplicate
    # integrated sections or figure cells.
    for index, cell in enumerate(nb.cells):
        source = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        if cell.cell_type == "markdown" and source.startswith("## 6. Saved preprocessing and repeated signatures"):
            nb.cells = nb.cells[:index]
            break
    _set_existing_markdown(nb, 0, """# Internet Advertisements: Big Data Hackathon

**Final integrated analysis notebook.** It documents data cleaning/selection/transformation, association mining, supervised classification, clustering, PySpark implementation status, plots, and reproducibility.

- **Dataset:** [UCI Internet Advertisements](https://archive.ics.uci.edu/dataset/51/internet%2Badvertisements)
- **Target:** `class` (`ad` / `nonad`)
- **Raw data:** the original UCI files are preserved in `data/raw/`; the cleaned copy is separate and keeps missing values.
- **Important limits:** the data-derived `origurl → ancurl → url` pattern analysis uses an analyst-defined structural order, not event chronology. The source has no timestamps or sessions. Both PySpark workloads executed on `local[2]`; Spark was slower than the matched local implementations on this small dataset.

The handout requires a readable PDF and source files. The final LaTeX report is being prepared separately from the saved analysis outputs.""")
    _set_existing_markdown(nb, 1, """## Assignment scope and execution plan

The WhatsApp handout asks for preprocessing, at least ten distinct plots with inferences, Apriori association mining (including the sequential-pattern wording), Decision Tree and DBSCAN analysis, CHARM closed itemsets/rules, two PySpark techniques, a local/Spark performance comparison, and a final PDF with source files.

This notebook reports executed values. A data-derived structural pattern analysis orders URL-named feature families, but the dataset has no timestamps, sessions, or natural event order; temporal sequence mining therefore remains unsupported. The synthetic GSP illustration is also separate. Both PySpark workloads executed in local mode and were slower than the matched local baselines.

The analysis builds on the raw-data checkpoint, corrected target-separated association outputs, grouped train/test tree artifacts, DBSCAN sensitivity output, and completed matched local/PySpark measurements in `artifacts/`.""")
    _set_existing_markdown(nb, 13, """### Cleaning and feature-selection decisions

- Source labels are normalized and feature values are parsed according to `ad.names`; no source row or raw file is changed.
- Keep unknown values as missing in the audit and saved cleaned copy. The 15 missing `local` values are not converted to observed zeroes.
- The Decision Tree imputes continuous fields with training-partition medians and binary fields with training-partition modes. The test partition never fits its preprocessing.
- DBSCAN uses only the three geometry fields, median-imputed on the unlabeled geometry table for this descriptive unsupervised run, then `log1p` and IQR scaling before Euclidean distance.
- Primary Apriori/CHARM transactions include only feature items. Class tokens are added in separate runs for class-consequent rules.
- Preserve duplicates. Group identical full feature signatures before the supervised split; keep conflicting-label signatures and account for them.
- Only constant features would be removed; the raw audit found no constant inputs. The 18-item association alphabet is a support filter for tractable mining, not a predictive feature ranking.""")
    _set_existing_markdown(nb, 14, """## 5. Descriptive plots

The original audit cells show class balance, missingness, raw geometry histograms, class-wise boxplots, width-height scatter, binary-indicator prevalence, and geometry correlation. The integrated numbered plot index below adds course forms and task diagnostics; each has its own actual-data interpretation.""")
    _set_existing_markdown(nb, 16, """### Visualization interpretation

The report figure manifest contains a specific interpretation for every figure. It includes count/proportion views; missingness and feature distribution plots; association-rule and CHARM summaries; holdout confusion/ROC/tree views; and DBSCAN parameter/cluster views. Supplementary diagnostics are labeled separately from the course forms.""")
    _set_existing_markdown(nb, 17, """## Integrated checkpoint

The original descriptive audit has been extended with target-separated association analysis, a grouped Decision Tree evaluation, feature-only DBSCAN, measured local/PySpark comparisons, individual plot interpretations, and reproducibility details. The separate PDF report and source bundle are built from these same artifacts.""")

    _append_integrated_sections(nb)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": sys.version.split()[0]}
    nb.metadata["execution"] = {"mode": "sequential in-process capture", "reason": "local socket permission denied to standard Jupyter kernel startup"}
    nbformat.validate(nb)
    execute_in_process(nb)
    nbformat.write(nb, NOTEBOOK_PATH)
    return {"notebook": str(NOTEBOOK_PATH), "cells": len(nb.cells), "executed_code_cells": sum(c.cell_type == "code" for c in nb.cells), "figure_count": len(json.loads((PROJECT_DIR / "artifacts/report/figure_manifest.json").read_text()))}


if __name__ == "__main__":
    print(json.dumps(build_notebook(), indent=2))
