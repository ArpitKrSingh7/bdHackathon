# Big Data Hackathon Progress

**Last updated:** 2026-11-21 (Asia/Kolkata)

**Delivery state:** an earlier notebook, ReportLab draft PDF, and source bundle were assembled and inspected. The audit, feature-selection sensitivity, schema-family sequence analysis, and synthetic Spark scale sweep through 500,000 rows were added afterward, so the PDF and ZIP are stale. No final LaTeX report or refreshed source bundle has been produced. Temporal sequential mining remains unresolved.

## Working rules

- `AGENTS.md` is the canonical project instruction file.
- Keep `data/raw/` unchanged; derive all results from the checked-in UCI files.
- Keep repeated feature signatures as repeated rows. Group identical signatures during supervised splitting so copies do not leak across train/test.
- Record only executed metrics, plots, rules, and timings. Update this file at each finished milestone.
- Keep the sequential-pattern demonstration separate from Internet Ads findings because the dataset has no timestamps, sessions, or event order.

## Completed: input audit and descriptive notebook

- The raw files `ad.data`, `ad.names`, and `ad.DOCUMENTATION` were byte-compared with the official UCI zip. SHA256 manifest is written in model metadata. No raw input has been changed.
- `hackathon_utils.py` loads the source data and computes feature signatures without imputation. The current notebook, `internet_ads_hackathon.ipynb`, has its original seven code cells executed in one Python process and saved outputs. A normal Jupyter kernel could not start because sandbox local-socket access was denied; this execution method is documented.
- Raw parse: 3,279 rows; 1,558 input features (3 continuous geometry fields and 1,555 binary); class counts 2,820 nonad / 459 ad. Missing values: height 903, width 901, aratio 910, local 15; 920 rows have at least one missing input. Exact duplicate rows including target: 856. Repeated feature signatures affect 1,137 rows (278 groups), and 3 groups have conflicting labels across 54 rows.
- All input fields vary. Geometry skewness is height 2.468, width 1.625, aratio 7.031. The UCI documentation's class counts differ by one from the checked-in data; this project uses the parsed raw-file counts.
- `scripts/data_quality_audit.py` executed a reproducible data-quality audit and saved `artifacts/data_quality_audit.json`. It verified all 1,555 binary fields are restricted to {0,1}, finite and positive geometry, 920 rows missing any input (910 missing geometry, 15 missing local), 856 exact duplicates (859 feature-only), 1,137 rows across 278 repeated-signature groups, 3 conflicting-label groups (54 rows), and 363 rows flagged by at least one geometry Tukey IQR fence (84 height, 282 width, 33 aratio). IQR flags are review flags, not proof of corruption; no rows are removed based on IQR alone.
- The first notebook checkpoint had 12 descriptive panels and seven chart forms. `report_figures.py` now generates 20 evidence-backed figures in 19 chart forms, each with a data-derived caption. The integrated notebook rebuilt and executed all 35 code cells, with the 20-figure interpretation index and refreshed Spark/MLflow outputs.

## Completed: Apriori, CHARM, and sequential-pattern treatment

Executed source: `association_mining.py`; outputs: `artifacts/association/`.

- Transactions are rows; duplicate signatures remain distinct transactions. The three geometry measurements are excluded from item mining; missing `local` is represented as `local=unknown`. Class is appended only for class-association rules.
- Primary feature-only mining uses min support 0.05 (164 rows) and min confidence 0.60: Apriori found 1,194 frequent itemsets / 44,122 rules in 0.01165 s; CHARM found 41 closed feature itemsets in 0.000264 s. Both class-augmented rule runs are separate: Apriori found 1,190 class-consequent rules from 2,388 augmented itemsets in 0.03741 s; CHARM found 37 class rules from 80 augmented closures in 0.000492 s. Both CHARM runs passed exhaustive TID-set verification.
- Sensitivity counts are internally labeled: feature-only itemset counts and separately class-augmented rule counts use max_len=3, whereas the primary runs have no maximum length. The 2% sensitivity row reports 531 class rules; it is not directly comparable with the primary unlimited-length total.
- The five-sequence GSP demo remains synthetic. A separate `scripts/schema_role_sequential_mining.py` now mines the actual 3,279 rows using the URL-named feature families in the analyst-defined order `origurl → ancurl → url`. It found 71 patterns and 29 prefix-extension rules at 5% support, 60% confidence, and max length 3. This is structural cross-field order, not observed chronology; it does not fulfill a temporal/session sequence requirement without instructor acceptance.

## Completed: Decision Tree and DBSCAN

Executed source: `classification_clustering.py`; outputs: `artifacts/models/`.

- Split: 2,623 train / 656 test, with zero shared feature-signature groups; ad prevalence was 13.99% overall, 13.99% train, 14.02% test. Continuous medians and binary modes were fit on training rows only. Repeated rows were retained.
- CART settings: Gini, depth 5, minimum leaf 20, balanced class weights, random seed 42. Test metrics: accuracy 0.9192, balanced accuracy 0.8848, ad precision 0.6696, recall 0.8370, F1 0.7440, ROC-AUC 0.9116; confusion matrix (actual nonad/ad by predicted nonad/ad) `[[526, 38], [15, 77]]`. Twelve IF-THEN leaf rules were extracted. The rule probability field was corrected to `weighted_predicted_class_probability_at_leaf` and defined as the predicted class share of the weighted leaf mass. In the latest tree-module run, train-only preprocessing, tree fit, and prediction measured 0.43610 s, 0.02920 s, and 0.00185 s.
- DBSCAN uses geometry only (height, width, aspect ratio), with log1p, median imputation, RobustScaler/IQR scaling, and Euclidean distance. The unlabeled geometry table was used for this descriptive clustering run; class labels were excluded from fit. A 10-nearest-neighbor distance-curve knee selected eps=0.457798; min_samples=10 was chosen for three dimensions and checked against 6/15 plus epsilon multipliers.
- Observed DBSCAN result: 3 clusters, 70 noise rows (2.13%), sampled non-noise silhouette 0.4991. Post-hoc ARI vs class was -0.0240 overall and -0.0176 excluding noise, indicating little agreement with the class labels at the selected setting. At epsilon ×0.75, the sensitivity sweep found 5 clusters/117 noise and post-hoc ARI 0.4062; this parameter sensitivity is reported, while the selected setting remains based on the feature-only k-distance knee. Sensitivity and noise tables/plots are saved under `artifacts/models/`.
- Decision Tree grouped cross-validation was executed via `scripts/grouped_cv_decision_tree.py` and saved under `artifacts/models/grouped_cv/grouped_cv_metrics.json` and `grouped_cv_per_fold.csv`. Across 5-fold GroupKFold × 3 repeats (15 total folds), identical feature signatures were strictly isolated to a single partition (zero group leakage verified), and imputation was fitted strictly inside each training fold. Results: accuracy 0.9328 ± 0.0128, balanced accuracy 0.8796 ± 0.0324, ad precision 0.7377 ± 0.0775, ad recall 0.8057 ± 0.0646, ad F1 0.7676 ± 0.0562, ROC-AUC 0.9002 ± 0.0256. The existing holdout results remain unchanged.
- `scripts/preprocessing_model_sensitivity.py` added a fixed, training-only ANOVA selector comparison on the same grouped holdout. SelectKBest(k=100) raised accuracy from 0.9192 to 0.9421, balanced accuracy from 0.8848 to 0.8935, and ad F1 from 0.7440 to 0.8000; recall changed from 0.8370 to 0.8261 and ROC-AUC from 0.9116 to 0.9094. This is one holdout sensitivity result, not a tuned final model. Scores and the top-20 plot are saved under `artifacts/models/preprocessing_sensitivity/`.
- The same script created `artifacts/processed/internet_ads_conflict_filtered_sensitivity.csv`, excluding all three complete exact-feature groups with inconsistent labels (54 rows: 48 nonad/1 ad in the largest group, plus groups of 2 and 3 rows). The original data and primary metrics remain unchanged. This target-informed variant is retrospective and must be presented only as a cleaning sensitivity case. The 363 IQR-flagged rows remain because no invalid geometry values were found and IQR alone does not establish corruption.

## Completed: saved preprocessing artifacts and report figures

- `preprocess_dataset.py` executed successfully and saved `artifacts/processed/internet_ads_cleaned.csv`, `preprocessing_config.json`, and `feature_signature_ledger.csv`. The cleaned copy preserves all 3,279 rows and missing values. It records 2,420 distinct signatures, 278 repeated groups (1,137 rows), and 3 conflicting-label groups; no duplicate row was removed.
- `report_figures.py` generated 20 figures in 19 distinct forms under `artifacts/report/figures/`, with per-figure captions in `figure_manifest.json`. The set documents ten course forms: bar, pie, histogram, dot, stem-and-leaf, scatter, box, decision-tree diagram, ROC curve, and a separately labeled geometry-only dendrogram; other association/model/runtime plots are supplementary diagnostics. Each figure has its own data-derived caption. All 20 figures are integrated into the 35-cell notebook and draft report; unmanifested file `19_geometry_dendrogram_sample.png` was removed. Final comprehensive LaTeX report will be compiled once the analysis pass is fully locked.
- Fixed the sanity-review tree probability label in the code/CSV/JSON/TXT to `weighted_predicted_class_probability_at_leaf`, with its weighted-leaf definition.
- Corrected `baseline_models_evaluation.md`: the unexecuted tree draft uses `fillna(-1)` and drops its first feature column without a stated reason, so the table no longer says there was no cleaning or selection. Refreshed `team_handoff.md` to match current association/Spark artifacts.
- `log_experiments.py` refreshed local MLflow with four final runs in experiment `Internet Ads Hackathon (final artifacts)` (experiment ID 2) after the target-separation and probability-label corrections. SQLite and artifact files remain under `artifacts/mlflow.db` and `artifacts/mlflow/`; no server is required.

## Completed: PySpark benchmarks and comparison

- `pyspark_benchmark.py` completed scikit-learn CART vs PySpark `DecisionTreeClassifier` on the same grouped split/train-only transformed arrays and local Apriori vs Spark FP-Growth on the same 3,279 row transactions and thresholds. PySpark 3.5.3 ran in `local[2]` with 24 CPUs visible; Spark startup took 1.5253 s and is kept separate from task timing.
- Decision Tree: local median fit 0.030632 s and predict 0.001710 s; Spark median fit 0.865233 s and predict/collect 0.104288 s. Spark DataFrame conversion/materialization took 4.725530 s. End-to-end Spark task time excluding startup was 5.699972 s, about 176.2× local fit+predict; fit alone was 28.25× slower. Local/Spark held-out accuracy was 0.9192/0.9207.
- Pattern mining: local Apriori took 0.037974 s; Spark FP-Growth fit took 0.071203 s and its conversion/materialization and collection took 0.112531 s and 0.261614 s. End-to-end Spark time excluding startup was 0.445911 s, about 11.74× the local Apriori runtime. Both returned 2,388 itemsets with matching supports. Spark emitted 11,709 singleton-consequent rules, matching the corresponding local subset; local's 133,556 total includes multi-item consequents. These are different algorithms and a task-level timing comparison. Spark was slower on this small local workload.
- `log_experiments.py` has refreshed the local MLflow experiment (ID 2) with four current runs, including completed Spark parameters, metrics, conversion/collection overhead, output counts, and slowdown ratios. The Spark benchmark run ID is `26edb63aa6b84dcc84adcc01d8654485`; tracking uses SQLite plus local artifacts, without a server.
- Refactored `build_report.py` prose to eliminate unevidenced theoretical linear scaling / cluster throughput claims, strictly keeping the observed single-machine `local[2]` overhead measurements.
- Synthetic row-replication scaling benchmark (`scripts/scaled_tree_spark_benchmark.py`) successfully executed end-to-end for both local scikit-learn CART and PySpark MLlib DecisionTree in `local[2]` across 7 sizes: 3,279, 10,000, 25,000, 50,000, 100,000, 250,000, and 500,000 training rows (saved to `artifacts/spark/scaled_tree_benchmark.json`, `.csv`, and `.png`). Observed results:
  - Local CART fit scaled from 0.0075 s (3.2k) to 0.1436 s (100k) and 0.9313 s (500k).
  - Spark CART fit scaled from 1.8755 s (3.2k) to 2.6942 s (100k) and 11.0962 s (500k).
  - Spark end-to-end (vector construction + DataFrame materialization + fit) scaled from 4.0438 s (3.2k) to 5.7900 s (100k) and 25.3173 s (500k).
  - **No crossover was observed up to 500,000 rows on a single machine.** Local Cython CART fit remained 11.9× to 250× faster than Spark MLlib fit, and 26.8× to 513× faster end-to-end. The relative gap narrowed at higher row counts (fixed Spark JVM/Py4J overhead amortizes), but local in-memory execution retained a consistent speed advantage. True scaling crossover would require multi-node cluster distribution or workloads exceeding single-machine RAM limits.
  - Each size has one timing observation, so this is an exploratory curve, not a stable scaling law. A separate 1,000,000-row attempt failed during Spark DataFrame cache/materialization with `java.lang.OutOfMemoryError: Java heap space`, before model fitting; no 1M timing exists. The successful 500k JSON/CSV/plot remain the benchmark artifacts. The raw 1M stderr is not yet archived, and 2M/4M were not tested. Do not publish extrapolated timings or assert that those sizes were measured.

## Completed: PDF, notebook, and source bundle

- `build_notebook.py` completed: `internet_ads_hackathon.ipynb` has 76 cells, 35 executed code cells, and the refreshed completed Spark measurements. The notebook uses sequential in-process execution because a standard Jupyter kernel could not create its local socket; every code cell completed during the builder run.
- `build_report.py` generated `internet_ads_hackathon_report.pdf` (17 A4 pages; 1,799,716 bytes; SHA-256 `231b965e09ee45843bd97aa7c242dd2a0e443aca77911a7818ce172939045b71`). Every page and all 20 figure panels were visually inspected in contact sheets; the detailed Spark comparison page and representative figures were inspected at full page size. One blank trailing page was removed and the report was regenerated. The tables, plots, sequential limitation, measured Spark slowdown, and source references are readable.
- The notebook builder was made rerunnable without duplicating generated sections, then rebuilt to 76 cells/35 executed code cells. It includes the reproducible package-build command and refreshed outputs.
- `internet_ads_source_bundle.zip` was assembled and inspected: required source/notebook/PDF/manifest/Spark outputs are included, along with the local MLflow SQLite database and artifact runs; raw data is intentionally excluded. `artifacts/report/source_bundle_manifest.json` records the final bundle size/hash and file inventory.
- All requested possible deliverables are now present. Dataset-level sequential mining remains explicitly infeasible as described above; the toy GSP demonstration is not substituted for it.
- Dataset-level sequential mining remains explicitly infeasible: this file has no timestamps, sessions, or natural event order. Five synthetic sequences produced 14 demonstration patterns; that toy GSP output is separate and does not satisfy or provide findings for the dataset-level requirement. Ordered event data or instructor-approved guidance would be needed.

## Key source files

- `internet_ads_hackathon.ipynb` — integrated notebook with 35 executed code cells.
- `hackathon_utils.py`, `preprocess_dataset.py`, `association_mining.py`, `classification_clustering.py`, `pyspark_benchmark.py`, `report_figures.py`, `experiment_tracking.py`, `log_experiments.py` — reusable analysis and tracking code.
- `scripts/data_quality_audit.py`, `scripts/grouped_cv_decision_tree.py`, `scripts/audit_dataset.py` — reproducible audit and cross-validation scripts.
- `scripts/schema_role_sequential_mining.py`, `scripts/preprocessing_model_sensitivity.py`, `scripts/scaled_tree_spark_benchmark.py` — dataset-derived non-temporal pattern analysis, train-only feature-selection/cleaning sensitivity, and a labeled scale benchmark.
- `report_latex/` — full publication-grade LaTeX submission draft (`main.tex`, `references.bib`, 23 figures, `asset_manifest.json`, `report_guide.md`) ready for Overleaf compilation.
- `artifacts/association/`, `artifacts/models/`, `artifacts/spark/`, `artifacts/models/grouped_cv/` — generated outputs.
- `requirements.txt` — observed package versions.
- `baseline_models_evaluation.md` — existing work reviewed against the raw source; its unsupported counts/claims were corrected in that working document before use.

Dataset source: [UCI Internet Advertisements](https://archive.ics.uci.edu/dataset/51/internet%2Badvertisements), [downloadable archive](https://archive.ics.uci.edu/static/public/51/internet+advertisements.zip).
