# Team Handoff — Internet Ads Hackathon

**Status:** delivery artifacts are assembled and inspected. Dataset-level sequential mining remains infeasible and is reported as an unfulfilled requirement, not as completed analytics. `AGENTS.md` is the project protocol and `progress.md` is the live checkpoint log.

## Current workspace snapshot

- Raw UCI files remain unchanged in `data/raw/`. The saved cleaned copy keeps all 3,279 rows and missing values; preprocessing configuration and the row-level feature-signature ledger are in `artifacts/processed/`.
- Feature-only Apriori found 1,194 itemsets/44,122 rules; feature-only CHARM found 41 closed itemsets. Separate class-augmented runs found 1,190 Apriori class rules and 37 CHARM class rules. Both CHARM exhaustive checks passed. The synthetic GSP output is not an Internet Ads finding.
- Decision Tree metrics use a grouped holdout with train-only preprocessing. The probability field is `weighted_predicted_class_probability_at_leaf`, defined as predicted-class share of class-weighted training mass. DBSCAN uses geometry only and reports noise/sensitivity under `artifacts/models/`.
- `report_figures.py` has regenerated 20 figures in 19 chart-form labels, each with an actual-data interpretation. The notebook now contains 76 cells, including 35 executed code cells, and its numbered plot index and Spark comparison reflect the saved benchmark.
- Both PySpark workloads completed using PySpark 3.5.3 with `local[2]`. Startup was 1.5253 s. Decision Tree Spark end-to-end was 5.7000 s excluding startup versus local fit+predict 0.03234 s (~176.2×). Spark FP-Growth end-to-end was 0.44591 s excluding startup versus local Apriori 0.03797 s (~11.74×). Spark was slower on this small local dataset.
- Association itemsets/supports matched (2,388 each). Spark emitted 11,709 singleton-consequent rules; the same local Apriori subset matches, while local's 133,556 total includes multi-item consequents. The rule totals therefore use different consequent scopes.
- `baseline_models_evaluation.md` accurately records the baseline's `fillna(-1)` and unexplained first-predictor exclusion instead of claiming no cleaning/selection.
- MLflow uses local SQLite and artifact storage; experiment ID 2 has four current runs, including Spark run `26edb63aa6b84dcc84adcc01d8654485`.
- `internet_ads_hackathon_report.pdf` has been generated as a 17-page report and visually reviewed across every page and figure; the detailed benchmark comparison page was checked at full page size.
- `internet_ads_source_bundle.zip` passes archive integrity inspection, includes the PDF/notebook/source files/figure manifest/Spark outputs and local MLflow store, and excludes raw data. `artifacts/report/source_bundle_manifest.json` records its hash and inventory.

## Ownership and boundaries

| Work area | Owner | Current state |
|---|---|---|
| Notebook integration, preprocessing, figures, report/PDF, source bundle, `progress.md`, this handoff | Primary integrator | In progress |
| `association_mining.py` and `artifacts/association/` | Association worker | Target-separated implementation and outputs handed off |
| `classification_clustering.py` and `artifacts/models/` | Classification/clustering worker | Tree probability-label correction and outputs handed off |
| `pyspark_benchmark.py` and `artifacts/spark/` | Spark worker | Both workloads executed; measured comparison shows slowdown |
| `sanity_review.md` | Sanity reviewer | Read-only findings; corrections tracked by the integrator |

Do not modify raw data. Keep the target out of primary feature-only Apriori/CHARM itemsets; use class-augmented transactions only for separate class-rule runs. Keep repeated rows as distinct transactions and split identical feature signatures as groups in supervised evaluation.

## Handoff status

1. The notebook was regenerated to 76 cells/35 executed code cells with no captured errors; Spark/MLflow outputs are integrated.
2. The PDF was visually checked across all pages and all 20 figure panels; the source ZIP was inspected and verified.
3. Dataset-level sequential mining remains infeasible because the data has no timestamps, sessions, or natural event order. The five-sequence synthetic GSP demonstration yields 14 patterns but is not a dataset analysis and does not fulfill this requirement.
4. Do not claim dataset-level sequential mining was completed. The generated report and notebook state the limitation and the need for ordered event data or instructor guidance.
