# Analysis changes to tell as a story

Use this as source material for the later LaTeX report. Describe each change as a response to evidence, and keep the original analysis visible for comparison. The new experiments are supplementary; they do not overwrite the original holdout or turn limitations into completed requirements.

## 1. From using every attribute to a train-only selection comparison

The first Decision Tree used all 1,558 inputs. A later review asked whether a smaller task-relevant feature set could retain performance while making the model easier to inspect. We added a fixed ANOVA filter: remove features constant in the training partition, then select the top 100 by `f_classif`. The selector is fitted only on the training data and evaluated on the same grouped holdout.

| Metric | All 1,558 inputs | Train-only top 100 | Change |
|---|---:|---:|---:|
| Accuracy | 0.9192 | 0.9421 | +0.0229 |
| Balanced accuracy | 0.8848 | 0.8935 | +0.0087 |
| Ad precision | 0.6696 | 0.7755 | +0.1059 |
| Ad recall | 0.8370 | 0.8261 | -0.0109 |
| Ad F1 | 0.7440 | 0.8000 | +0.0560 |
| ROC-AUC | 0.9116 | 0.9094 | -0.0022 |

The selected model makes fewer false ad alerts on this holdout (38 to 22) and has one fewer detected ad (77 to 76). This is a precision/recall trade-off, not a universal improvement claim; it is one fixed-k comparison, not cross-validated feature-selection tuning. Evidence: `scripts/preprocessing_model_sensitivity.py` and `artifacts/models/preprocessing_sensitivity/`.

## 2. From a normalized copy to an explicit cleaning decision

The normalized copy preserves 3,279 rows and unknown values. The reproducible audit found no invalid binary domains and no non-finite or non-positive observed geometry. It did find 920 rows with at least one missing input, 363 rows flagged by a geometry IQR fence, and three exact full-feature signatures with contradictory labels across 54 rows.

The revision creates a separate file excluding those complete conflicting-label groups for sensitivity analysis. It does not overwrite raw data or the primary model. The largest ambiguous group has 48 nonad labels and 1 ad label; the other groups have 1/1 and 2/1 label counts. These records are ambiguous, but the available data cannot identify which label is correct. The 363 IQR flags stay in the primary data because a fence is not proof of corruption. The audit's unsupported 728×90 example was removed; the raw dataset's maximum width is 640.

Be precise in the report: an ambiguity-filtered sensitivity copy exists; the project did not delete all IQR-flagged rows or establish that those rows were noise. Do not claim the entire preprocessing requirement is unquestionably complete if the marker interprets “remove noisy data” as mandatory row deletion.

## 3. From five hand-written sequences to patterns from actual rows

The original toy GSP fixture remains an algorithm example only. A new analysis uses observed binary indicators from the `origurl`, `ancurl`, and `url` feature families in all 3,279 rows. Under the explicitly analyst-defined order `origurl → ancurl → url`, it found 71 patterns and 29 prefix-extension rules at 5% support, 60% confidence, and length at most three.

This improves the work from a toy-only illustration to data-derived structural patterns. It still is not chronology: the source contains field-family indicators, not raw URLs, sessions, timestamps, or event order. Call it schema-family ordered analysis and mark temporal sequential mining unresolved unless the instructor accepts that definition. Evidence: `scripts/schema_role_sequential_mining.py` and the `schema_role_*` outputs under `artifacts/association/`.

## 4. From one small Spark comparison to a scale-test attempt

The executed Internet Ads comparison remains the primary platform result: Spark `local[2]` was slower on 3,279 rows (about 176.2× end-to-end for the tree and 11.74× for pattern mining). I then ran the repeated-row tree benchmark from 3,279 through 500,000 synthetic training rows in `local[2]`. At 500k, local fit was 0.9313 s and Spark fit 11.0962 s; expansion-plus-fit was 0.9429 s local versus 25.3173 s Spark, excluding 1.5384 s startup. Across the tested sizes, Spark remained slower, but the gap narrowed from about 249.7× to 11.9× on fit and from about 513.3× to 26.8× end-to-end. Each size has only one timing observation. These are real measurements of a synthetic repeated-row workload, not new Internet Ads evidence or a stable scaling law.

My first attempt from the restricted shell could not open the Py4J socket; the completed sweep was run locally. I also attempted 1M rows, but the Spark DataFrame cache/materialization step failed with Java heap exhaustion before tree fitting. There is no 1M timing, and I did not test 2M/4M. The raw failure stderr is not currently archived; preserve it before citing the failure as reproducible evidence. Repeated rows add no independent observations; label the successful sweep a synthetic throughput stress test and keep it separate from real-data findings. It remains single-machine `local[2]`, not a cluster test. Each point has one timing observation, so avoid claims about a stable crossover.

## 5. Submission narrative and current limits

The story is: inspect and validate the data; choose task-specific representations; add a train-only feature-selection comparison; investigate conflicting labels without silently replacing the original data; mine actual field-family patterns while stating the order assumption; and compare Spark honestly. The remaining mark risks are temporal sequence compliance, a Spark speedup that has not been demonstrated, and literal row-noise removal if required by the marker.

No new PDF or LaTeX report was generated. The existing ReportLab draft and source bundle predate these additions and need a final refresh after the technical choices are locked and the GitHub URL is known.
