"""Build a readable PDF report from executed analysis artifacts."""
from __future__ import annotations

import hashlib
import json
import xml.sax.saxutils as sax
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.utils import ImageReader

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS = PROJECT_DIR / "artifacts"
OUTPUT_PDF = PROJECT_DIR / "internet_ads_hackathon_report.pdf"
CONTENT_WIDTH = A4[0] - 76
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

for font_name, font_path in (("DejaVu", FONT_REGULAR), ("DejaVu-Bold", FONT_BOLD), ("DejaVu-Mono", FONT_MONO)):
    if Path(font_path).is_file():
        pdfmetrics.registerFont(TTFont(font_name, font_path))
    else:
        font_name = "Helvetica"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _esc(value) -> str:
    return sax.escape(str(value))


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontName="DejaVu-Bold", fontSize=24, leading=30, alignment=TA_CENTER, textColor=colors.HexColor("#17324D"), spaceAfter=12))
    styles.add(ParagraphStyle(name="SubtitleCustom", parent=styles["Normal"], fontName="DejaVu", fontSize=11, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#43566B"), spaceAfter=8))
    styles.add(ParagraphStyle(name="SectionCustom", parent=styles["Heading1"], fontName="DejaVu-Bold", fontSize=16, leading=20, textColor=colors.HexColor("#17324D"), spaceBefore=8, spaceAfter=8, keepWithNext=True))
    styles.add(ParagraphStyle(name="SubsectionCustom", parent=styles["Heading2"], fontName="DejaVu-Bold", fontSize=11.5, leading=15, textColor=colors.HexColor("#245A76"), spaceBefore=8, spaceAfter=5, keepWithNext=True))
    styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontName="DejaVu", fontSize=8.8, leading=12.5, spaceAfter=6, alignment=TA_LEFT, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="SmallCustom", parent=styles["BodyText"], fontName="DejaVu", fontSize=7.2, leading=9.2, spaceAfter=2, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="CaptionCustom", parent=styles["BodyText"], fontName="DejaVu", fontSize=8.0, leading=11, textColor=colors.HexColor("#263746"), spaceBefore=3, spaceAfter=8, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="TableHeaderCustom", parent=styles["BodyText"], fontName="DejaVu-Bold", fontSize=7.2, leading=9, textColor=colors.white, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="TableCellCustom", parent=styles["BodyText"], fontName="DejaVu", fontSize=7.0, leading=8.7, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="CalloutCustom", parent=styles["BodyText"], fontName="DejaVu-Bold", fontSize=9.2, leading=13, borderColor=colors.HexColor("#D3A24C"), borderWidth=0.8, borderPadding=8, backColor=colors.HexColor("#FFF8E8"), spaceBefore=6, spaceAfter=9, wordWrap="CJK"))
    return styles


STYLES = _build_styles()


def p(text: str, style: str = "BodyCustom") -> Paragraph:
    return Paragraph(_esc(text), STYLES[style])


def rich(text: str, style: str = "BodyCustom") -> Paragraph:
    return Paragraph(text, STYLES[style])


def heading(text: str, level: int = 1) -> Paragraph:
    return p(text, "SectionCustom" if level == 1 else "SubsectionCustom")


def _table(data, widths, *, long=False, header=True, font_size=7.0):
    prepared = []
    for row_index, row in enumerate(data):
        prepared.append([
            Paragraph(_esc(cell), STYLES["TableHeaderCustom" if header and row_index == 0 else "TableCellCustom"])
            for cell in row
        ])
    cls = LongTable if long else Table
    tbl = cls(prepared, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D5DDE4")),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#284F6B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ])
    for row_index in range(1 if header else 0, len(data)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F4F7F9")))
    tbl.setStyle(TableStyle(commands))
    return tbl


def _image(path: Path, max_width=CONTENT_WIDTH, max_height=4.2 * inch):
    width, height = ImageReader(str(path)).getSize()
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale, hAlign="CENTER")


def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DejaVu", 7)
    canvas.setFillColor(colors.HexColor("#627487"))
    if doc.page > 1:
        canvas.drawString(38, A4[1] - 24, "Internet Advertisements | Big Data Hackathon")
    canvas.drawRightString(A4[0] - 38, 20, f"Page {doc.page}")
    canvas.restoreState()


def build_report() -> dict:
    assoc = load_json(ARTIFACTS / "association" / "association_metadata.json")
    model = load_json(ARTIFACTS / "models" / "classification_clustering_summary.json")
    spark = load_json(ARTIFACTS / "spark" / "pyspark_benchmark_summary.json")
    prep = load_json(ARTIFACTS / "processed" / "preprocessing_config.json")
    manifest = load_json(ARTIFACTS / "report" / "figure_manifest.json")
    tree = model["decision_tree"]
    dbscan = model["dbscan"]
    feature_mining = assoc["feature_only_mining"]
    class_mining = assoc["class_association_rule_mining"]

    class_rules = pd.read_csv(ARTIFACTS / "association" / "apriori_class_rules.csv")
    charm_class_rules = pd.read_csv(ARTIFACTS / "association" / "charm_class_rules.csv")
    sensitivity = pd.read_csv(ARTIFACTS / "association" / "apriori_support_sensitivity.csv")
    tree_rules = pd.read_csv(ARTIFACTS / "models" / "decision_tree_if_then_rules.csv")
    dbscan_sensitivity = pd.read_csv(ARTIFACTS / "models" / "dbscan_parameter_sensitivity.csv")
    cluster_profile = pd.read_csv(ARTIFACTS / "models" / "dbscan_cluster_class_profile.csv")

    styles = STYLES
    story = []
    story.append(Spacer(1, 0.55 * inch))
    story.append(p("Internet Advertisements", "TitleCustom"))
    story.append(rich("<b>Big Data Hackathon — final analysis report</b>", "SubtitleCustom"))
    story.append(rich("Prepared 21 November 2026<br/>Source data: UCI Machine Learning Repository, Internet Advertisements<br/>Submission contents: executed notebook, PDF report, commented source bundle, and generated outputs", "SubtitleCustom"))
    story.append(Spacer(1, 0.28 * inch))
    story.append(rich("<b>Delivery status:</b> the local analytics, notebook, visualizations, PDF, and source bundle are delivered. Both PySpark workloads completed and were slower than the matched local tasks on this small dataset. Dataset-level sequential mining is infeasible because the file has no event order; the separate synthetic GSP demonstration is not a dataset result.", "CalloutCustom"))

    story.append(heading("Executive summary"))
    summary_rows = [
        ["Area", "Executed finding / status"],
        ["Data", "3,279 rows; 1,558 input features (3 continuous geometry fields, 1,555 binary); 2,820 nonad and 459 ad. Missingness and repeated signatures are retained and documented."],
        ["Apriori / CHARM", "Feature-only Apriori: 1,194 itemsets and 44,122 rules. Feature-only CHARM: 41 closed itemsets, exhaustive TID-set verification passed. Separate class-rule runs produced 1,190 Apriori and 37 CHARM class rules."],
        ["Decision Tree", f"Grouped 80/20 holdout; accuracy {tree['accuracy']:.3f}, ad F1 {tree['f1_ad']:.3f}, ROC-AUC {tree['roc_auc']:.3f}; 12 IF-THEN rules."],
        ["DBSCAN", f"Geometry-only Euclidean DBSCAN: {dbscan['clusters_excluding_noise']} clusters, {dbscan['noise_count']} noise rows ({dbscan['noise_fraction']:.2%}); sampled non-noise silhouette {dbscan['silhouette_non_noise_sampled']:.3f}."],
        ["Visualizations", f"{len(manifest)} interpreted figures in {len(set(item['chart_type'] for item in manifest))} chart forms, including 10 course-aligned forms and separately labeled supplementary diagnostics."],
        ["Sequential requirement", "Dataset-level sequence mining not possible from the available cross-sectional fields. A synthetic GSP demo is clearly separate and does not count as Internet Ads findings."],
        ["PySpark", f"Decision Tree and FP-Growth completed on local[2]. Spark/local end-to-end ratios: {spark['tree_timing_comparison']['spark_end_to_end_excluding_startup_seconds'] / (spark['local_tree']['fit_seconds_median'] + spark['local_tree']['predict_seconds_median']):.1f}× for tree and {spark['arm_timing_comparison']['spark_fp_end_to_end_excluding_startup_seconds'] / spark['local_apriori']['elapsed_seconds']:.1f}× for pattern mining; Spark was slower."],
    ]
    story.append(_table(summary_rows, [120, CONTENT_WIDTH - 120]))

    story.append(heading("1. Dataset audit and integrity"))
    story.append(p("The assignment handout names the UCI Internet Advertisements dataset. We parsed `ad.data` using the feature declarations in `ad.names`, treated `?` as missing, trimmed the trailing period from labels, and converted features to numeric values. The raw files were retained unchanged."))
    raw_manifest = prep["source"]["raw_sha256_manifest"]
    raw_rows = [["Source file", "Bytes", "SHA-256"]]
    for item in raw_manifest:
        raw_rows.append([item["file"], f"{item['bytes']:,}", item["sha256"]])
    story.append(_table(raw_rows, [115, 55, CONTENT_WIDTH - 170]))
    story.append(Spacer(1, 6))
    audit_rows = [
        ["Audit item", "Observed from the checked-in raw data"],
        ["Shape / features", "3,279 rows; 1,558 input features; 3 continuous geometry fields and 1,555 binary indicators."],
        ["Target", "2,820 nonad and 459 ad; 14.00% ad. UCI page documentation differs by one example (2,821 / 458), so the raw file is the calculation source."],
        ["Missing values", "height 903; width 901; aratio 910; local 15. At least one geometry value is missing in 910 rows; at least one input is missing in 920 rows."],
        ["Exact duplicate rows", "856 duplicate rows including target."],
        ["Repeated full-feature signatures", "2,420 distinct signatures; 278 repeated groups contain 1,137 rows (859 excess rows); 3 groups have conflicting labels across 54 rows."],
        ["Association binary signatures", "Because geometry is excluded, the binary-only representation is coarser: 1,976 unique binary signatures; 278 repeated groups contain 1,659 rows (1,303 excess rows). Every row still remains a separate transaction."],
    ]
    story.append(_table(audit_rows, [130, CONTENT_WIDTH - 130]))
    story.append(p("All 1,558 input columns vary in the raw file, so the general cleaned copy drops no features. The association task selects 18 binary items at 5% item support for the primary mining alphabet; a sensitivity sweep uses a common alphabet chosen at 2%."))

    story.append(heading("2. Preprocessing and transformations"))
    prep_rows = [
        ["Task", "Representation / cleaning decision", "Leakage or target handling"],
        ["Saved cleaned copy", "Normalized labels/types; missing entries unchanged; all 3,279 rows retained.", "No imputation in this general copy."],
        ["Decision Tree", "Median imputation for continuous fields; mode imputation for binary fields; no scaling.", "Fit only on the training partition. Full feature signatures are grouped before split."],
        ["Apriori / CHARM", "One row is one transaction; observed binary 1 values are items; geometry excluded; missing local gets `local=unknown`.", "Primary itemsets/rules are feature-only. Class tokens are used only in separately labeled class-rule runs."],
        ["DBSCAN", "Geometry median imputation, log1p, RobustScaler (IQR), Euclidean distance.", "Descriptive unlabeled geometry run; class excluded from fit and used only for post-hoc profiling."],
    ]
    story.append(_table(prep_rows, [86, 255, CONTENT_WIDTH - 341]))
    story.append(p("The saved files `artifacts/processed/internet_ads_cleaned.csv`, `preprocessing_config.json`, and `feature_signature_ledger.csv` make the processing choices and repeated-row accounting inspectable."))

    story.append(heading("3. Association mining: Apriori and CHARM"))
    min_support = feature_mining["apriori_parameters"]["min_support"]
    min_count = feature_mining["apriori_parameters"]["min_support_count"]
    min_conf = feature_mining["apriori_parameters"]["min_confidence"]
    association_rows = [
        ["Run", "Transactions/items", "Observed outputs", "Mining time"],
        ["Feature-only Apriori", f"{assoc['rows_as_transactions']:,} rows; {assoc['selected_binary_feature_count']} selected binary items; support {min_support:.0%} ({min_count}), confidence {min_conf:.0%}.", f"{feature_mining['apriori_frequent_itemset_count']:,} frequent itemsets; {feature_mining['apriori_rule_count_at_min_confidence']:,} rules.", f"{feature_mining['apriori_runtime_seconds_excluding_transaction_build']:.6f} s"],
        ["Feature-only CHARM", "Same row-preserving feature-only transactions and support threshold.", f"{feature_mining['charm_closed_itemset_count']} closed itemsets; exhaustive verification matched={feature_mining['charm_exhaustive_verification']['matches_exhaustive']}.", f"{feature_mining['charm_runtime_seconds_excluding_transaction_build']:.6f} s"],
        ["Apriori class rules", "Separate transactions append `class=ad` / `class=nonad`.", f"{class_mining['apriori_class_rule_count_at_min_confidence']:,} single-class-consequent rules from {class_mining['apriori_itemset_count']:,} augmented frequent itemsets.", f"{class_mining['apriori_runtime_seconds_excluding_transaction_build']:.6f} s"],
        ["CHARM class rules", "Separate class-augmented CHARM run; labels not in primary feature closures.", f"{class_mining['charm_class_rule_count_at_min_confidence']} class rules from {class_mining['charm_class_augmented_closed_itemset_count']} augmented closures; exhaustive verification matched={class_mining['charm_exhaustive_verification']['matches_exhaustive']}.", f"{class_mining['charm_runtime_seconds_excluding_transaction_build']:.6f} s"],
    ]
    story.append(_table(association_rows, [95, 160, 205, CONTENT_WIDTH - 460]))
    story.append(heading("Example rules", 2))
    top_rule = class_rules.sort_values(["lift", "confidence"], ascending=False).iloc[0]
    story.append(p(f"The highest-lift Apriori class rule is `{top_rule['antecedent_text']} -> {top_rule['consequent_text']}`: support count {int(top_rule['support_count'])} ({top_rule['support']:.3%}), confidence {top_rule['confidence']:.3%}, lift {top_rule['lift']:.3f}. This is an association in this dataset, not a causal claim."))
    feature_top = pd.read_csv(ARTIFACTS / "association" / "apriori_rules.csv").sort_values(["lift", "confidence"], ascending=False).iloc[0]
    story.append(p(f"The top feature-only Apriori rule is `{feature_top['antecedent_text']} -> {feature_top['consequent_text']}`: support count {int(feature_top['support_count'])} ({feature_top['support']:.3%}), confidence {feature_top['confidence']:.3%}, lift {feature_top['lift']:.3f}."))
    top_charm = charm_class_rules.sort_values(["lift", "confidence"], ascending=False).iloc[0]
    story.append(p(f"The highest-lift CHARM class rule is `{top_charm['antecedent_text']} -> {top_charm['consequent_text']}` with confidence {top_charm['confidence']:.3%} and lift {top_charm['lift']:.3f}. Per Lectures 10 and 11, Closed Frequent Itemsets (CFIs) provide a lossless compression: all 1,194 frequent itemsets and their exact supports can be recovered from the 41 closures via the closure operator f(x) = i(t(x)). Unlike Maximal Frequent Itemsets (MFIs, Lecture 12), CFIs preserve exact support counts."))
    story.append(heading("Support sensitivity", 2))
    sensitivity_rows = [["Support", "Minimum rows", "Feature itemsets ≤3", "Class rules ≤3", "Feature runtime (s)", "Class runtime (s)"]]
    for _, row in sensitivity.iterrows():
        sensitivity_rows.append([f"{row['min_support']:.0%}", str(int(row["minimum_count"])), str(int(row["feature_itemsets_through_length_3"])), str(int(row["class_rules_at_confidence_0_60_max_len_3"])), f"{row['feature_only_runtime_seconds']:.6f}", f"{row['class_augmented_runtime_seconds']:.6f}"])
    story.append(_table(sensitivity_rows, [55, 62, 95, 95, 95, CONTENT_WIDTH - 402]))
    story.append(p("The sensitivity sweep caps union itemset size at three and therefore is not directly comparable with the primary unlimited-length runs."))

    story.append(heading("Sequential-pattern requirement"))
    toy = assoc["toy_sequential_demo"]
    story.append(rich("<b>Dataset-level sequential mining is infeasible with the available columns.</b> Internet Ads has no timestamps, sessions, or natural event order. Row order and feature-column order are not treated as time. The saved GSP-style demonstration uses " + str(toy["sequence_count"]) + " explicitly synthetic sequences and is kept separate from all dataset findings; it does not fulfill sequential mining on Internet Ads.", "CalloutCustom"))
    story.append(p(f"Synthetic demo settings only: support {toy['parameters']['min_support']:.0%} ({toy['parameters']['min_support_count']} of {toy['sequence_count']} toy sequences), max pattern length {toy['parameters']['max_len']}, and {toy['pattern_count']} patterns. See `toy_sequential_patterns.csv`."))

    story.append(heading("4. Decision Tree classification"))
    split = tree["split"]
    story.append(p(f"A grouped holdout selected 2,623 training rows and 656 test rows. Identical full feature signatures have zero overlap across partitions. Ad prevalence is {split['class_prevalence_train']:.2%} in train and {split['class_prevalence_test']:.2%} in test. Continuous medians and binary modes were fitted on training rows only. Settings: Gini criterion, max depth {tree['tree_parameters']['max_depth']}, minimum leaf {tree['tree_parameters']['min_samples_leaf']}, balanced class weights, random seed {tree['tree_parameters']['random_state']}."))
    metrics_rows = [
        ["Metric", "Held-out result"],
        ["Accuracy", f"{tree['accuracy']:.4f}"],
        ["Balanced accuracy", f"{tree['balanced_accuracy']:.4f}"],
        ["Ad precision", f"{tree['precision_ad']:.4f}"],
        ["Ad recall / Sensitivity (TPR)", f"{tree['recall_ad']:.4f}"],
        ["Specificity (TNR)", "0.9326"],
        ["False positive rate (FPR)", "0.0674"],
        ["Ad F1", f"{tree['f1_ad']:.4f}"],
        ["ROC-AUC", f"{tree['roc_auc']:.4f}"],
        ["Confusion matrix (actual nonad/ad rows; predicted nonad/ad columns)", "[[526, 38], [15, 77]]"],
    ]
    story.append(_table(metrics_rows, [330, CONTENT_WIDTH - 330]))
    story.append(p("The weighted class-probability field in the rule outputs is `weighted_predicted_class_probability_at_leaf`: the predicted class's share of class-weighted training mass at that leaf. It is not an unweighted ad fraction."))
    story.append(heading("Extracted IF-THEN leaf rules", 2))
    rule_rows = [["#", "IF conditions", "THEN", "Train rows", "Weighted predicted-class probability"]]
    for _, row in tree_rules.iterrows():
        rule_rows.append([str(int(row["rule_id"])), str(row["if"]), str(row["then"]), str(int(row["training_rows_at_leaf"])), f"{row['weighted_predicted_class_probability_at_leaf']:.3f}"])
    story.append(_table(rule_rows, [24, 250, 42, 58, CONTENT_WIDTH - 374], long=True))

    story.append(heading("5. DBSCAN clustering"))
    story.append(p(f"DBSCAN uses the three geometry fields only. Their different scales and skew motivated median imputation, `log1p`, and RobustScaler/IQR scaling before Euclidean distance. Epsilon was selected by the maximum perpendicular-distance knee on the sorted 10-nearest-neighbor distance curve; min_samples=10 exceeds the d+1 heuristic for three dimensions and was compared with 6 and 15."))
    dbscan_rows = [
        ["Output", "Observed value"],
        ["epsilon / min_samples", f"{dbscan['eps_selected']:.6f} / {dbscan['min_samples_selected']}"],
        ["Clusters excluding noise", str(dbscan["clusters_excluding_noise"])],
        ["Noise", f"{dbscan['noise_count']} / {dbscan['rows_clustered']} ({dbscan['noise_fraction']:.2%})"],
        ["Sampled non-noise silhouette", f"{dbscan['silhouette_non_noise_sampled']:.4f}"],
        ["Post-hoc ARI vs class, all rows", f"{dbscan['adjusted_rand_vs_class_all_rows_posthoc']:.4f}"],
        ["Post-hoc ARI vs class, non-noise", f"{dbscan['adjusted_rand_vs_class_non_noise_posthoc']:.4f}"],
    ]
    story.append(_table(dbscan_rows, [240, CONTENT_WIDTH - 240]))
    story.append(p("Class labels were not used for imputation, scaling, distance, or fitting. The ARI values are post-hoc only. At epsilon×0.75 the sweep produced five clusters, 117 noise rows, and post-hoc all-row ARI 0.4062, while the selected feature-only knee setting produced ARI -0.0240. This sensitivity is why epsilon was not selected by class alignment."))
    profile_rows = [["Cluster label", "Ads", "Nonads", "Rows"]]
    for _, row in cluster_profile.iterrows():
        cluster_label = str(row["cluster"])
        if cluster_label.endswith(".0"):
            cluster_label = cluster_label[:-2]
        profile_rows.append([cluster_label, str(int(row["ad"])), str(int(row["nonad"])), str(int(row["All"]))])
    story.append(heading("Cluster/noise profile", 2))
    story.append(_table(profile_rows, [120, 100, 100, CONTENT_WIDTH - 320]))
    story.append(p("All noise-point rows and the epsilon/min_samples sensitivity matrix are included in the source bundle under `artifacts/models/`. Additionally, to address the assignment prompt's explicit requirement for dendrograms in clustering, Figure 20 presents an Agglomerative Hierarchical Clustering dendrogram (Lecture 21) using complete linkage and Euclidean distance (L2 norm, Lecture 22), illustrating cluster merge heights."))

    story.append(heading("6. PySpark implementations and local measurements"))
    local_tree = spark["local_tree"]
    local_apriori = spark["local_apriori"]
    spark_tree = spark["spark_tree"]
    spark_fp = spark["spark_fpgrowth"]
    tree_e2e = spark["tree_timing_comparison"]["spark_end_to_end_excluding_startup_seconds"]
    fp_e2e = spark["arm_timing_comparison"]["spark_fp_end_to_end_excluding_startup_seconds"]
    local_tree_e2e = local_tree["fit_seconds_median"] + local_tree["predict_seconds_median"]
    tree_ratio = tree_e2e / local_tree_e2e
    fp_ratio = fp_e2e / local_apriori["elapsed_seconds"]
    spark_rows = [
        ["Task", "Local", "PySpark", "Spark / local"],
        ["Decision Tree fit", f"CART median fit {local_tree['fit_seconds_median']:.4f} s", f"Spark median fit {spark_tree['fit_seconds_median']:.4f} s", f"{spark_tree['fit_seconds_median'] / local_tree['fit_seconds_median']:.1f}×"],
        ["Tree fit + prediction", f"Fit + predict {local_tree_e2e:.4f} s", f"End-to-end (conversion + fit + predict/collect) {tree_e2e:.4f} s", f"{tree_ratio:.1f}×"],
        ["Association fit", f"Apriori {local_apriori['elapsed_seconds']:.4f} s", f"FP-Growth median fit {spark_fp['fit_seconds_median']:.4f} s", f"{spark_fp['fit_seconds_median'] / local_apriori['elapsed_seconds']:.1f}×"],
        ["Association end-to-end", f"Apriori {local_apriori['elapsed_seconds']:.4f} s", f"FP-Growth (conversion + fit + collect) {fp_e2e:.4f} s", f"{fp_ratio:.1f}×"],
    ]
    story.append(_table(spark_rows, [105, 125, 190, CONTENT_WIDTH - 420]))
    story.append(rich(f"<b>Observed result: both PySpark workloads completed, and Spark was slower on this 3,279-row local workload.</b> Spark 3.5.3 ran with master `local[2]` and 24 CPUs visible. Its startup took {spark['spark_startup_seconds']:.3f} s and is excluded from the task times above. Tree conversion/materialization took {spark_tree['spark_dataframe_conversion_and_materialization_seconds']:.3f} s, followed by {spark_tree['fit_seconds_median']:.3f} s median fit and {spark_tree['predict_and_collect_seconds_median']:.3f} s median prediction/collection. FP-Growth conversion/materialization took {spark_fp['spark_dataframe_conversion_and_materialization_seconds']:.3f} s, followed by {spark_fp['fit_seconds_median']:.3f} s median fit and {spark_fp['collect_itemsets_and_rules_seconds_median']:.3f} s median collection. These measurements include Spark-side data preparation; local tree inputs were already transformed arrays.", "CalloutCustom"))
    story.append(p(f"The Decision Tree used the same grouped train/test split and train-only transformed arrays; local and Spark held-out accuracies were {local_tree['metrics']['accuracy']:.4f} and {spark_tree['metrics']['accuracy']:.4f}, respectively. The association task used the same {local_apriori['transaction_count']:,} row transactions and support/confidence thresholds. Both returned {local_apriori['frequent_itemset_count']:,} frequent itemsets with matching supports. Spark emitted {spark_fp['association_rule_count']:,} singleton-consequent rules; local Apriori emitted {local_apriori['association_rule_count']:,} rules when multi-item consequents were allowed. The 11,709 singleton-consequent local rules match Spark's rule set, so the larger all-rule count is a scope difference, not rules missed by Spark. Apriori and FP-Growth are different algorithms; this is a task-level timing comparison, not an implementation-equivalence claim."))
    story.append(p("Performance context: For this 3,279-row dataset running in local[2] mode, local scikit-learn execution was faster than PySpark for both tasks. Spark's coordination, serialisation, and JVM startup impose fixed overhead that dominates at this scale. Whether PySpark would be faster on larger datasets or multi-node clusters was not tested and is not claimed here; such a comparison would require an actual distributed benchmark with measured results."))

    story.append(heading("7. MLflow tracking"))
    experiment_name = "Internet Ads Hackathon (final artifacts)"
    story.append(p(f"MLflow 3.3.2 is installed. The final tracking configuration uses local SQLite at `artifacts/mlflow.db` and local artifact files at `artifacts/mlflow/`, without starting a server. The experiment is `{experiment_name}`. It logs parameters, actual metrics, and output files, including the completed Spark runs and their measured timings."))
    story.append(p("MLflow is useful here as a compact experiment log during the hackathon, but the executed notebook, source bundle, and this report remain the submission record."))

    story.append(PageBreak())
    story.append(heading("8. Visualization inventory and interpretations"))
    course_forms = [
        "bar chart", "pie chart", "histogram", "dot plot", "stem-and-leaf display", "scatter plot", "box plot",
        "decision tree diagram", "ROC curve", "dendrogram (supplementary course visualization)",
    ]
    story.append(p("At least ten distinct course-aligned forms are present: " + "; ".join(course_forms) + ". The dendrogram is a fixed-seed geometry-only sample illustration, not a DBSCAN result. Supplementary figures provide missingness, association-rule, confusion-matrix, DBSCAN, and measured local-versus-Spark runtime diagnostics."))
    story.append(p(f"The executed figure manifest contains {len(manifest)} numbered figures and {len(set(item['chart_type'] for item in manifest))} chart-form labels. Each has a title/axes and an individual data-derived interpretation."))

    for index, item in enumerate(manifest, start=1):
        image_path = PROJECT_DIR / item["file"]
        group = [
            heading(f"Figure {index}. {item['chart_type']}", 2),
            _image(image_path),
            p(item["caption"], "CaptionCustom"),
        ]
        story.append(KeepTogether(group))
        if index != len(manifest):
            story.append(Spacer(1, 7))
        if index % 2 == 0 and index != len(manifest):
            story.append(PageBreak())

    story.append(PageBreak())
    story.append(heading("9. Conclusions, scope limits, and delivery status"))
    story.append(p("The executed supervised model identifies a useful fraction of ads on a leakage-controlled, signature-grouped holdout. The association results show recurring feature combinations and class rules; their support/confidence/lift values describe co-occurrence rather than causal effects. DBSCAN groups the geometry representation but its alignment with class labels changes with epsilon."))
    story.append(rich("<b>Dataset-level sequential mining remains infeasible from this dataset.</b> The file has no timestamps, sessions, or natural event order, so its rows and feature columns cannot be turned into sequences without inventing chronology. The GSP-style output is a separate synthetic demo and does not fulfill dataset-level sequential mining. Both PySpark techniques did execute and were measured; on this small local workload, Spark was slower than the matched local tasks.", "CalloutCustom"))
    story.append(p("Completing the sequential-pattern requirement would require ordered event/session data or an instructor-approved alternative. No temporal dataset findings are claimed. The raw data can be downloaded from the official UCI archive."))

    story.append(heading("Reproducibility and source files"))
    files = [
        "internet_ads_hackathon.ipynb — integrated notebook with executed outputs",
        "hackathon_utils.py — validated UCI loader and raw-file manifest",
        "preprocess_dataset.py — cleaned copy, preprocessing config, signature ledger",
        "association_mining.py — Apriori, CHARM, separate synthetic GSP demo",
        "classification_clustering.py — grouped Decision Tree and DBSCAN analyses",
        "pyspark_benchmark.py — PySpark Decision Tree / FP-Growth and matched local baselines",
        "report_figures.py — 20 interpreted figure outputs",
        "experiment_tracking.py and log_experiments.py — local MLflow setup and final run records",
        "build_notebook.py, build_report.py, package_sources.py — reproducible deliverable builders",
        "requirements.txt, progress.md, team_handoff.md, artifacts/ — versions, checkpoint history, generated outputs",
    ]
    for item in files:
        story.append(p("• " + item))
    story.append(p("Source bundle: `internet_ads_source_bundle.zip` (the raw UCI data is referenced but not duplicated in the bundle)."))
    story.append(heading("References", 2))
    reference_html = [
        '<link href="https://archive.ics.uci.edu/dataset/51/internet%2Badvertisements" color="blue">UCI Internet Advertisements dataset page and citation</link>',
        '<link href="https://archive.ics.uci.edu/static/public/51/internet+advertisements.zip" color="blue">UCI downloadable archive</link>',
        '<link href="https://doi.org/10.1137/1.9781611972726.27" color="blue">Zaki and Hsiao (2002), CHARM: An Efficient Algorithm for Closed Itemset Mining</link>',
        '<link href="https://spark.apache.org/docs/3.5.3/api/python/reference/api/pyspark.ml.classification.DecisionTreeClassifier.html" color="blue">Apache Spark 3.5.3 DecisionTreeClassifier API</link>',
        '<link href="https://spark.apache.org/docs/3.5.3/api/python/reference/api/pyspark.ml.fpm.FPGrowth.html" color="blue">Apache Spark 3.5.3 FP-Growth API</link>',
        '<link href="https://mlflow.org/docs/latest/tracking/" color="blue">MLflow Tracking documentation</link>',
        "Course slides in the workspace: Lecture4 Data Representation Mechanisms; Lecture13 Data Classification; Lecture20 Clustering; Lecture21 Hierarchical Clustering.",
        "Assignment source: `WhatsApp Image 2026-11-21 at 11.33.37 AM.jpeg`.",
    ]
    for line in reference_html:
        if line.startswith("<link"):
            story.append(rich(line))
        else:
            story.append(p(line))

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF), pagesize=A4,
        rightMargin=38, leftMargin=38, topMargin=42, bottomMargin=34,
        title="Internet Advertisements Big Data Hackathon Report",
        author="Big Data Hackathon Team",
        subject="Executed analytics, visualizations, and reproducibility status",
    )
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    digest = hashlib.sha256(OUTPUT_PDF.read_bytes()).hexdigest()
    return {
        "pdf": str(OUTPUT_PDF),
        "bytes": OUTPUT_PDF.stat().st_size,
        "sha256": digest,
        "figures": len(manifest),
        "course_aligned_forms_listed": len(course_forms),
        "pages": None,
        "spark_status": spark["status"],
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
