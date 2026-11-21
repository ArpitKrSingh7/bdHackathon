# Independent Sanity Review — Internet Ads Hackathon

**Review date:** 2026-11-21  
**Scope:** Assignment handout, end-semester lecture slides, saved baseline artifacts, the updated audit, and the three new sensitivity/sequence/scale scripts and outputs. The existing PDF and ZIP predate these latest additions.

## Verdict

The core analysis remains reproducible, and the latest review added a train-only feature-selection comparison, a transparent ambiguity-filtered sensitivity copy, and a dataset-derived structural pattern analysis. It did **not** change raw data or replace the original model results. The existing 17-page ReportLab PDF and ZIP were checked earlier, but they predate this latest work and are now stale. No final LaTeX report has been generated.

**I would not call this fully rubric-compliant yet.** True temporal sequential mining is not supported by the source data. The measured Spark workloads on the real 3,279-row dataset were slower. A larger synthetic Spark extension is coded but could not start in this sandbox because Py4J socket creation is denied. The quality audit found no objectively invalid feature values; 54 rows with conflicting full-feature labels are isolated in a separate retrospective sensitivity file, while IQR flags are retained. Literal “remove noisy data” compliance therefore remains a judgment call for the marker.

## High-priority rubric risks

### HIGH — Dataset-level sequential-pattern requirement remains unfulfilled

The original GSP demo still uses five hand-authored synthetic sequences. A new script, `scripts/schema_role_sequential_mining.py`, also mines actual row indicators from the `origurl`, `ancurl`, and `url` feature families. Under the explicitly analyst-defined order `origurl → ancurl → url`, it found 71 patterns and 29 prefix-extension rules at 5% support, 60% confidence, and maximum length 3. This is structural field-family order, not observed event chronology. It is a better dataset-based exploration than the toy alone, but temporal sequential mining remains unfulfilled unless the instructor accepts this sequence definition. Do not call it a user journey or time sequence.

### HIGH — No PySpark performance enhancement was observed

The actual-data benchmark used Spark 3.5.3 `local[2]` on 3,279 rows and showed slowdowns: about 176.2× end-to-end for the Decision Tree and 11.74× for pattern mining versus local Apriori, with startup separated. The synthetic repeated-row tree benchmark covered seven sizes through 500,000 rows. Spark was slower at every size. At 500k, local fit was 0.9313 s and Spark fit 11.0962 s; local expansion-plus-fit was 0.9429 s versus 25.3173 s for Spark expansion/conversion/materialization/fit. Startup was 1.5384 s, separately measured. Ratios narrowed from 249.7× to 11.9× on fit and 513.3× to 26.8× end-to-end. Each point has one timing observation, and repeated rows add no independent evidence. This is a synthetic `local[2]` stress test, not a real-data speedup or cluster result. A separate 1M attempt failed before fitting with Java heap exhaustion during DataFrame materialization; the successful 500k JSON remains intact, but the 1M stderr is not archived. The 2M/4M sizes were not tested.

The prior PDF was already known to contain an unsupported cluster-scaling/OOM claim, while the current `build_report.py` source was softened later. The prior PDF remains a stale draft and must not be treated as final evidence.

**Additional report consistency issue:** the draft PDF's clustering narrative says the supplementary dendrogram uses complete linkage, while the plotted figure, `figure_manifest.json`, and `report_figures.py` identify average linkage. The current figure-generation code uses `method="average"`. The later LaTeX report must describe the generated figure accurately or regenerate the plot if the analysis is intentionally changed. The current `build_report.py` performance-context prose has since been softened, but the already-generated PDF is older and still contains the unsupported cluster/OOM claim.

### MEDIUM — Cleaning does not explicitly address noisy/invalid records

`scripts/data_quality_audit.py` now saves a reproducible audit. All 1,555 binary columns obey {0,1}; all observed geometry is finite and positive; 920 rows have missing inputs; and 363 rows trigger an IQR fence. These are unusual-value flags, not verified bad records, so the primary dataset retains them. Three exact full-feature groups (54 rows) have conflicting labels. `scripts/preprocessing_model_sensitivity.py` creates a separate CSV excluding those groups, with the target-informed retrospective-selection caveat documented. This is defensible ambiguity handling, but it does not prove which labels were wrong or remove IQR-flagged records. The handout's literal noise-removal wording remains a potential deduction. The earlier 728×90 example was removed because dataset maximum width is 640.

### MEDIUM — Feature selection was added as a sensitivity, not the default model

The original tree still uses all 1,558 features. A fixed `SelectKBest(f_classif, k=100)` comparison now fits feature selection only on the training side of the same grouped holdout. Accuracy changed 0.9192→0.9421, balanced accuracy 0.8848→0.8935, precision 0.6696→0.7755, recall 0.8370→0.8261, F1 0.7440→0.8000, and ROC-AUC 0.9116→0.9094. This is one holdout sensitivity result; it should not be called cross-validated or selected as final by testing the same holdout.

## Delivery and consistency findings

- The previously inspected PDF is 17 A4 pages and has no obvious blank/truncated page, but it predates the new analysis and is not the final LaTeX deliverable.
- The notebook has 76 cells; all 35 code cells have execution counts 1–35 and there are no captured error outputs.
- `artifacts/report/figure_manifest.json` continues to describe the 20 primary figures. The LaTeX preparation pack now has three supplementary figures: tree feature importance, training-only ANOVA feature scores, and the non-temporal URL-family pattern support plot. Its asset manifest includes hashes for these copies.
- `progress.md` now records the audit correction and new analysis. However, the previous source ZIP predates these changes and needs rebuilding after the technical work is locked. Raw data remains excluded from that earlier bundle.
- The ten claimed course-aligned forms are plausible against the available lectures: Lecture 4 covers pie, histogram, bar graph, dot plot, stem-and-leaf, scatter, and box plots; Lecture 13 covers Decision Trees and ROC; Lectures 20/21 cover dendrograms. The report correctly identifies the dendrogram as a separate geometry-only sample illustration, not a DBSCAN result.

## Verified earlier / saved-output checks

- Raw dataset audit, association outputs, CHARM exhaustive checks, class-rule separation, and Spark/local singleton-consequent equivalence were independently checked in the prior review pass and remain consistent with current saved artifacts.
- The current report correctly distinguishes 11,709 Spark singleton-consequent rules from local Apriori's 133,556 rules that include multi-item consequents; it does not present the difference as missing Spark rules.
- Tree metrics use a grouped split and train-only imputation. DBSCAN is described as geometry-only and parameter-sensitive.
- The selected 100-feature comparison is fitted without training/test leakage from the selector; its fixed-k result is saved separately from baseline outputs.
- The saved benchmark artifact contains successful measurements through 500,000 rows. The separate 1M attempt failed before fitting; it produced no timing, and 2M/4M were not tested.

## Reviewer status

**Artifacts:** baseline PDF/ZIP were inspected earlier; both predate the newest scripts and are stale for final submission. The LaTeX folder remains a preparation pack, not a compiled report.  
**Analysis:** new data audit, feature-selection sensitivity, schema-family pattern mining, and the seven-point Spark `local[2]` stress sweep through 500k were executed. Spark remained slower at all sizes. A 1M attempt failed before fit; no 2M/4M runs are recorded.  
**Sign-off:** not cleared for final submission. Obtain instructor guidance on sequence-definition acceptance and the Spark-performance rubric; preserve the 1M stderr if cited; refresh the final report and source bundle afterward.
