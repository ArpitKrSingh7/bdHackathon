# Final report handoff — LaTeX

**LaTeX draft is created.** `main.tex` and `references.bib` have been authored in this folder adhering to all specifications in `report_guide.md`. Upload this entire folder to Overleaf or compile locally with pdfLaTeX. Do not reuse the old ReportLab PDF or old source ZIP as the final version: both predate the latest analysis and figures.

## Author and dataset

- **Author:** Arpit Kumar Singh
- **Roll number:** CS23I1022
- **Course:** 25CS5004 / CS 5010 — Analytics & Systems of Big Data Practice
- **Dataset:** UCI Internet Advertisements, 3,279 rows and 1,558 input features in the checked-in raw file.
- **Dataset page:** <https://archive.ics.uci.edu/dataset/51/internet%2Badvertisements>
- **GitHub:** <https://github.com/ArpitKrSingh7/bdHackathon>

## Files in this folder

- `report_guide.md` — report story, verified values, caveats, section order, references, and LaTeX layout notes.
- `analysis_evolution.md` — baseline-to-revision narrative for the methods and honest limitations.
- `asset_manifest.json` — image paths, captions, roles, and checksums.
- `figures/` — 20 primary figures and 3 supplementary figures.

The 20 primary figures cover the EDA, association mining, CHARM, Decision Tree, DBSCAN, Spark comparison, and dendrogram. The supplementary images are Decision Tree feature importance, train-only ANOVA feature scores, and URL-family pattern support. Use paths relative to this folder, for example `figures/17_dbscan_k_distance.png`. The schema-pattern image is explicitly non-temporal; do not caption it as a visit sequence.

## Evidence to cite from the project

Use generated artifacts as the source of numeric claims. In particular, refresh against these before final compilation:

- `../artifacts/data_quality_audit.json` — schema validity, missingness, duplicates, conflicting labels, and IQR review flags.
- `../artifacts/models/preprocessing_sensitivity/preprocessing_sensitivity.json` and `../artifacts/models/preprocessing_sensitivity/anova_feature_scores.csv` — fixed top-100 feature-selection comparison; one grouped holdout only.
- `../artifacts/association/schema_role_sequence_metadata.json`, `schema_role_sequential_patterns.csv`, and `schema_role_sequential_rules.csv` — actual-row structural patterns under the analyst-defined `origurl → ancurl → url` order. They are not temporal sequences.
- `../artifacts/spark/pyspark_benchmark_summary.json` — actual 3,279-row PySpark benchmarks.
- `../artifacts/spark/scaled_tree_benchmark.json` and `.csv` — completed single-machine synthetic scale sweep through 500,000 rows. Each size has one timed observation; the repeated rows add no independent examples; Spark remained slower.
- `../artifacts/mlflow.db` and `../artifacts/mlflow/` — refresh run/parameter/metric counts immediately before writing the MLflow section.

I also attempted a 1,000,000-row run. Spark failed before model fitting during DataFrame caching/materialization with `java.lang.OutOfMemoryError: Java heap space`; therefore there is no 1M timing or performance result. I did not measure 2M or 4M rows. The raw stderr from the failed attempt is not currently saved with the benchmark JSON, so preserve it before citing the failure as archived evidence. Do not include extrapolated 1M/2M/4M times.

## Report direction

Tell the story in this order: dataset and question; data audit and task-specific preprocessing; EDA plots with individual inferences; Apriori/CHARM; the schema-family sequence experiment and its non-temporal limit; Decision Tree and the feature-selection comparison; DBSCAN; real-data and synthetic-scale Spark results; MLflow; combined findings, limitations, and reproducibility. The unresolved rubric risks are temporal sequence compliance and the absence of a measured Spark speedup. State these plainly and ask the instructor for interpretation where possible.

Use the `report_guide.md` for exact metrics, plot interpretations, and method caveats. Keep sentences accessible, vary their length, define technical terms before using them, and distinguish measured findings from assumptions. Include only code excerpts that correspond to the current source files. Do not generate final claims from the previous PDF or from the synthetic-only parts of the scale test.

## Overleaf handoff

Upload this whole folder so its relative `figures/` paths remain valid. Add `main.tex` and references, use `\includegraphics[width=\linewidth]{figures/...}`, and keep large plots on their own page when needed. Compile with ordinary pdfLaTeX-compatible packages; avoid shell-escape requirements. Before submission, verify page breaks, figure captions, references, URLs, and every reported value. The final PDF, source bundle, and GitHub repository must all reflect the same code revision.
