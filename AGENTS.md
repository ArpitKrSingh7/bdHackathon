# AGENTS.md — 25CS5004 / CS 5010 Big Data Endsem

## 1. Grounding & Anti-Hallucination Rules
- Never fabricate metrics, accuracy scores, rule counts, itemsets, execution timings, or plot observations.
- Every reported number and visualization inference must come from actual code execution or an inspected output artifact.
- Inspect the actual dataset files (`ad.names`, `ad.data`) before making assumptions about columns, feature types, missing values, class distribution, or encoding.
- Inspect the installed environment and package versions before assuming dependencies are available.
- If a file, package, column, or dataset field is missing or ambiguous, investigate it and explicitly document any assumption.
- Distinguish observed results from assumptions, algorithm parameters, and theoretical expectations.
- Do not claim that a script, algorithm, visualization, benchmark, or PDF succeeded unless execution produced the expected artifact without errors.

## 2. Scope & Deliverable Discipline
Stay focused on the official problem statement.
Required deliverables:
1. Data cleaning, selection, and transformation.
2. At least 10 unique and meaningful visualizations with individual interpretations.
3. Apriori association rule mining, including the sequential patterns requirement.
4. Decision Tree classification with extracted IF-THEN rules and appropriate performance metrics.
5. DBSCAN clustering with parameter justification and relevant evaluation.
6. CHARM algorithm for Closed Frequent Itemsets (CFIs) and relevant association rules.
7. PySpark implementations of at least two applicable predictive analytics techniques.
8. Performance comparison between appropriate local and PySpark implementations.
9. Commented source code, execution outputs, interpretations, and conclusions.
10. A comprehensive final PDF report and all required source files.

Do not introduce unrequested deep learning, gradient boosting, or unrelated algorithms.

## 3. Dataset Understanding & Preprocessing
- Preserve the original dataset in its raw form.
- Document dataset dimensions, feature types, target labels, and missing-value distribution.
- Clearly justify every cleaning and transformation decision.
- Document feature selection criteria and the reason for excluding any attributes.
- Distinguish raw values from imputed, normalized, encoded, or transformed values.
- Avoid data leakage, especially during train-test splitting and preprocessing.
- Preserve the target variable separately from unsupervised mining tasks unless its inclusion is explicitly justified.
- Save the processed dataset and preprocessing configuration for reproducibility.

## 4. Algorithmic Correctness
- Implement algorithms according to their actual definitions.
- Do not confuse ordinary Apriori association mining with sequential pattern mining.
- Define the transaction/sequence representation explicitly before applying sequential Apriori.
- State and justify parameters such as minimum support, minimum confidence, minimum lift, DBSCAN epsilon, and minimum samples.
- CHARM must generate closed frequent itemsets, not merely ordinary frequent itemsets.
- Extract understandable IF-THEN rules from the Decision Tree.
- For DBSCAN, report cluster labels, noise points, and suitable clustering evaluation measures.
- Explain algorithm limitations and parameter sensitivity wherever relevant.

## 5. PySpark & Performance Evaluation
- Implement at least two relevant techniques using PySpark.
- Compare against an appropriate non-distributed implementation using equivalent input data, preprocessing, and algorithm settings wherever possible.
- Record actual execution times and relevant resource/configuration details.
- Explain the hardware, Spark configuration, dataset size, and experimental setup.
- Distinguish Spark startup and initialization overhead from algorithm execution time.
- Never claim performance enhancement unless supported by measured results.
- If distributed execution is slower, report that honestly and explain possible reasons.

## 6. Visualization & Reporting
- Produce at least 10 genuinely distinct visualizations.
- Avoid duplicate plots with cosmetic changes.
- Every plot must include:
  - Meaningful title and axis labels.
  - Appropriate scales and legends.
  - Actual data-derived values.
  - A short interpretation explaining the observed pattern.
- Clearly distinguish observations from causal explanations.
- Include important parameters, outputs, and limitations in the final report.
- Ensure that tables and figures are readable in the exported PDF.

## 7. Reproducibility & Code Quality
- Use modular, reusable, clearly commented code.
- Keep raw data, processed data, source code, plots, metrics, and reports organized.
- Fix random seeds wherever applicable.
- Record dependency versions and execution configurations.
- Validate input data and handle errors explicitly.
- Avoid unnecessary duplication of code across notebooks and scripts.
- Ensure that the project can be executed using documented instructions.

## 8. Multi-Agent Coordination
- Treat the shared Git repository as the single source of truth.
- Assign clearly defined modules and responsibilities to individual agents.
- Do not independently overwrite or modify another agent's module without coordination.
- Follow agreed input/output interfaces between modules.
- Avoid duplicate implementations unless explicitly requested for independent verification.
- Review generated code before integration.
- Run integration tests after combining modules.

## 9. Final Verification Checklist
Before declaring the project complete:
- [ ] All dataset assumptions verified against original files.
- [ ] Preprocessing executed and outputs inspected.
- [ ] At least 10 unique visualizations generated and interpreted.
- [ ] Apriori and sequential pattern mining executed.
- [ ] Decision Tree trained, evaluated, and rules extracted.
- [ ] DBSCAN executed and clustering results inspected.
- [ ] CHARM closed frequent itemsets generated and verified.
- [ ] At least two PySpark techniques executed.
- [ ] Local vs. PySpark benchmarks measured and documented.
- [ ] All reported metrics verified against actual outputs.
- [ ] Source code and dependencies documented.
- [ ] Final PDF generated and visually inspected.
- [ ] All required source files included in the submission.

> **Core Principle:** Correctness, reproducibility, and evidence matter more than impressive-looking results.
