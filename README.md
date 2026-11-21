# Finding Patterns in Internet Advertisements: Exploratory, Predictive, and Big-Data Analytics

[![Python](https://img.shields.io/badge/Python-3.10.12-blue.svg)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5.3-orange.svg)](https://spark.apache.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7.1-green.svg)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-brightgreen.svg)](https://mlflow.org/)
[![Status](https://img.shields.io/badge/Status-Completed%20%26%20Verified-success.svg)](#)

**Course:** 25CS5004 / CS 5010 — Analytics & Systems of Big Data Practice (Endsem Practical Evaluation)  
**Institution:** Indian Institute of Information Technology, Design and Manufacturing (IIITDM), Kancheepuram  
**Student Name:** Arpit Kumar Singh  
**Roll Number:** `CS23I1022`  
**Instructor:** Dr. B. Sivaselvan  
**Dataset:** [UCI Internet Advertisements Benchmark (DOI: 10.24432/C5V011)](https://archive.ics.uci.edu/dataset/51/internet%2Badvertisements)

---

## Executive Summary & Deliverables

This repository contains an end-to-end, empirical big-data analytics project conducted on the **UCI Internet Advertisements dataset** ($N = 3{,}279$ instances, $1{,}558$ input attributes). Every reported metric, execution latency, rule count, and visualization is generated through executed code and inspected output artifacts. No figures or numbers are fabricated.

### Official Rubric Deliverable Matrix

| # | Handout Deliverable | Implementation & Code Entry Point | Observed Result / Status |
|---|:---|:---|:---|
| **1** | **Data Cleaning, Selection & Transformation** | [`scripts/data_quality_audit.py`](scripts/data_quality_audit.py), [`preprocess_dataset.py`](preprocess_dataset.py), [`scripts/preprocessing_model_sensitivity.py`](scripts/preprocessing_model_sensitivity.py) | Verified all 1,555 binary fields $\in \{0, 1\}$; 920 rows missing input; 856 exact duplicates; 363 Tukey IQR flags retained with justification; train-only median/mode imputation. |
| **2** | **$\ge 10$ Course Visualizations with Inferences** | [`report_figures.py`](report_figures.py), [`artifacts/report/figures/`](artifacts/report/figures/) | **20 figures across 19 forms** mapped directly to Lectures 4, 13, 20, 21. |
| **3** | **Apriori Association Rule Mining** | [`association_mining.py`](association_mining.py) | **1,194 frequent itemsets** ($44{,}122$ rules) on feature-only; **1,190 class rules** on class-augmented transactions ($s_{\min}=5\%, c_{\min}=60\%$). |
| **4** | **Sequential Pattern Requirement** | [`scripts/schema_role_sequential_mining.py`](scripts/schema_role_sequential_mining.py), [`artifacts/association/toy_sequential_patterns.csv`](artifacts/association/toy_sequential_patterns.csv) | **Limitation documented:** Dataset lacks timestamps/event order. Provided 5-sequence synthetic GSP demo + 3,279-row URL-family structural sequence mining (71 patterns, 29 rules). |
| **5** | **Decision Tree Classification & Rules** | [`classification_clustering.py`](classification_clustering.py), [`scripts/grouped_cv_decision_tree.py`](scripts/grouped_cv_decision_tree.py) | CART depth 5 on signature-aware holdout ($2{,}623/656$): **Accuracy 91.92%, Balanced Acc 88.48%, Ad-$F_1$ 0.7440, ROC-AUC 0.9116**, 12 IF-THEN rules. Verified via **15-fold Grouped CV** ($0.9328 \pm 0.0128$). |
| **6** | **DBSCAN Clustering & Parameter Tuning** | [`classification_clustering.py`](classification_clustering.py) | Evaluated on 3D geometry; 10-NN elbow knee at $\varepsilon=0.4578, \text{MinPts}=10$: **3 dense clusters, 70 noise rows ($2.13\%$), silhouette 0.4991**. Post-hoc ARI vs class is $-0.0240$. |
| **7** | **CHARM Closed Frequent Itemsets** | [`association_mining.py`](association_mining.py) | Discovered **41 feature-only CFIs** in $264\ \mu\text{s}$ and **80 class-augmented CFIs** ($492\ \mu\text{s}$, 37 class rules). Passed exhaustive TID-set intersection verification. |
| **8** | **PySpark Distributed Implementations** | [`pyspark_benchmark.py`](pyspark_benchmark.py) | Implemented PySpark MLlib `DecisionTreeClassifier` and `FPGrowth` on matched data in `local[2]` mode. |
| **9** | **Local vs. PySpark Performance Comparison** | [`pyspark_benchmark.py`](pyspark_benchmark.py), [`scripts/scaled_tree_spark_benchmark.py`](scripts/scaled_tree_spark_benchmark.py) | Real data: Spark is $28.2\times$ slower on fit. Synthetic scaling ($3.2\text{k} \to 500\text{k}$ rows): Local Cython CART remains $11.9\times$ to $250\times$ faster on fit; **no single-node crossover**. 1M row OOM failure boundary documented. |
| **10** | **Experiment Provenance & Tracking** | [`log_experiments.py`](log_experiments.py), [`experiment_tracking.py`](experiment_tracking.py) | Embedded SQLite database [`artifacts/mlflow.db`](artifacts/mlflow.db) tracking 8 runs, 50 parameters, 83 metrics, and 25 artifacts. |
| **11** | **Comprehensive LaTeX Report** | [`report_latex/main.tex`](report_latex/main.tex), [`report_latex/references.bib`](report_latex/references.bib) | Full publication-grade LaTeX report source with 23 embedded figures, ready for Overleaf compilation. |

---

## Repository Structure

```text
bdHackathon/
├── README.md                                 # GitHub repository documentation
├── requirements.txt                          # Pinned dependency environment
├── internet_ads_hackathon.ipynb              # 84-cell Jupyter notebook with 39 executed code cells
├── report.pdf                                # Complete 19-page compiled end-semester evaluation report
├── hackathon_utils.py                        # Reusable raw data loader & feature signature hasher
├── preprocess_dataset.py                     # Preprocessing pipeline and signature ledger exporter
├── association_mining.py                     # Dual-engine Apriori, CHARM, & structural sequence mining
├── classification_clustering.py              # Grouped-split CART decision tree & geometry DBSCAN
├── pyspark_benchmark.py                      # Matched Local vs. PySpark 3.5.3 benchmark runner
├── report_figures.py                         # 20-figure manifest generator across 19 chart forms
├── log_experiments.py                        # Local MLflow logging script (Experiment ID 2)
├── experiment_tracking.py                    # Lightweight SQLite MLflow wrapper
├── package_sources.py                        # Reproducible submission bundle archiver
├── data/
│   └── raw/                                  # Pristine, byte-verified UCI archive files
│       ├── ad.data                           # 3,279 rows × 1,559 columns
│       ├── ad.names                          # Feature schema definitions
│       └── ad.DOCUMENTATION                  # Original UCI documentation
├── scripts/
│   ├── data_quality_audit.py                 # Schema, domain, duplicate, & Tukey IQR audit
│   ├── grouped_cv_decision_tree.py           # 5-fold GroupKFold × 3 repeats (15 total folds)
│   ├── schema_role_sequential_mining.py      # Structural URL-family sequence miner (3,279 rows)
│   ├── preprocessing_model_sensitivity.py    # Train-only ANOVA top-100 feature selection
│   └── scaled_tree_spark_benchmark.py        # 7-size synthetic scaling sweep (3.2k to 500k rows)
├── artifacts/
│   ├── data_quality_audit.json               # Machine-readable schema & IQR audit outputs
│   ├── association/                          # Apriori & CHARM CSVs, metadata, and sequential outputs
│   ├── models/                               # Decision tree rules, metrics, grouped CV, & DBSCAN
│   ├── spark/                                # PySpark benchmark JSONs, CSVs, and scaling curves
│   ├── processed/                            # Cleaned data, preprocessing config, signature ledger
│   └── report/figures/                       # 20 publication figures (01_class_counts to 20_dendrogram)
└── report_latex/                             # Full LaTeX report source pack for Overleaf
    ├── main.tex                              # 11-section publication-ready report source
    ├── main.pdf                              # 19-page compiled publication report
    ├── references.bib                        # 18 complete bibliographic citations
    ├── asset_manifest.json                   # Checksums and metadata for 23 embedded figures
    └── figures/                              # 20 primary + 3 supplementary figures
```

---

## Data Intake, Schema & Quality Audit

The raw data consists of $3{,}279$ instances and $1{,}559$ attributes parsed from `data/raw/ad.data` and `data/raw/ad.names`.

### Verified Dataset Facts

| Property | Parsed & Verified Value | Description / Significance |
|---|:---|:---|
| **Instances** | $3{,}279$ rows | Total observations from web crawl (1998). |
| **Input Features** | $1{,}558$ features | 3 continuous geometry features + 1,555 binary indicators. |
| **Ground-Truth Target** | `nonad`: $2{,}820$ ($86.00\%$), `ad`: $459$ ($14.00\%$) | **Imbalance Ratio:** $6.14:1$. Raw accuracy alone is misleading. *(Note: UCI documentation cites 2,821 / 458; our calculations strictly use the parsed file).* |
| **Binary Indicators** | $1{,}555$ columns checked | **0 violations.** 100% strictly restricted to $\{0, 1\}$. Sparsity is $98.98\%$ zeros. |
| **Missingness** | $920$ rows ($28.06\%$) have missing input | `height`: 903 ($27.54\%$), `width`: 901 ($27.48\%$), `aratio`: 910 ($27.75\%$), `local`: 15 ($0.46\%$). |
| **Geometric Range** | Finite and positive | $\text{height} \in [1, 640]$, $\text{width} \in [1, 640]$, $\text{aratio} \in [0.0015, 60.0]$. |
| **Exact Duplicates** | $856$ rows | Identical inputs and identical target class. |
| **Unique Signatures** | $2{,}420$ distinct feature vectors | 278 signatures repeat across $1{,}137$ rows. |
| **Conflicting Labels** | $3$ signature groups ($54$ rows) | Group 1: 48 nonad / 1 ad; Group 2: 2 ad / 1 nonad; Group 3: 1 ad / 1 nonad. |
| **Tukey IQR Flags** | $363$ rows ($11.07\%$) | 84 height, 282 width, 33 aspect ratio. |

> **Methodological Justification for Outlier Retention:**  
> Standard $1.5 \times \text{IQR}$ fences flag extreme values because web advertising geometry possesses severe right skewness ($\text{aratio moment skew} = 7.0264$). Industry display advertising legitimately uses $1 \times 1$ tracking pixels as well as wide horizontal banners up to $640\text{ px}$. Deleting these 363 rows would eliminate valid commercial advertisements. Instead, continuous dimensions are processed using robust transformations ($\log(1+x)$ and `RobustScaler`).

---

## Exploratory Data Analysis & Visualizations

We produced **20 figures across 19 distinct forms**, directly mapping to the graphical representations in course lectures:
- **Lecture 4 (Data Representation Mechanisms):** Bar chart, pie chart, histogram, dot plot, stem-and-leaf, scatter plot, box plot.
- **Lecture 13 (Classification):** Decision tree architecture, ROC curve.
- **Lectures 20 & 21 (Clustering):** Spatial projection scatter, average-linkage dendrogram.

### 1. Class Distribution & Missingness Topology

| Figure 1: Class Frequencies (Lecture 4 Bar Chart) | Figure 2: Class Proportions (Lecture 4 Pie Chart) |
|:---:|:---:|
| ![Figure 1](artifacts/report/figures/01_class_counts.png) | ![Figure 2](artifacts/report/figures/02_class_share.png) |
| *2,820 nonads vs. 459 ads ($6.14:1$ imbalance). Requires class-aware metrics.* | *Ads constitute only $14.0\%$ of the corpus; majority voting would fail ad detection.* |

| Figure 3: Missingness Across Features (Lollipop) | Figure 4: Continuous Geometry Densities (Histogram) |
|:---:|:---:|
| ![Figure 3](artifacts/report/figures/03_missingness_lollipop.png) | ![Figure 4](artifacts/report/figures/04_geometry_histograms.png) |
| *Missingness is isolated to geometry ($\approx 27.5\%$) and local ($0.46\%$).* | *$\log(1+x)$ reveals multimodality across banner dimensions (raw skew: 2.47, 1.62, 7.03).* |

---

### 2. Geometric Diagnostics & Bivariate Relationships

| Figure 5: Geometric Spread by Class (Box Plot) | Figure 6: Dimension Correlation (Scatter Plot) |
|:---:|:---:|
| ![Figure 5](artifacts/report/figures/05_geometry_boxplots.png) | ![Figure 6](artifacts/report/figures/06_width_height_scatter.png) |
| *Ad median width is $460\text{ px}$ vs. $101\text{ px}$ for nonads; interquartile ranges overlap.* | *Spearman rank correlation $\rho = 0.411$ ($p < 10^{-15}$). Substantial dispersion observed.* |

| Figure 8: Correlation Heatmap | Figure 9: Height Stem-and-Leaf (Lecture 4 Sample) |
|:---:|:---:|
| ![Figure 8](artifacts/report/figures/08_geometry_correlation_heatmap.png) | ![Figure 9](artifacts/report/figures/09_height_stem_leaf.png) |
| *Spearman correlations: height/width $0.41$, height/aratio $-0.51$, width/aratio $0.49$.* | *Sample of 100 values: heavy concentration in the 10s and 20s (19 in stem 1).* |

| Figure 10: Width Distribution Shape (Violin Plot) | Figure 11: Aspect Ratio Empirical CDF |
|:---:|:---:|
| ![Figure 10](artifacts/report/figures/10_width_violin.png) | ![Figure 11](artifacts/report/figures/11_aratio_ecdf.png) |
| *Ad widths peak sharply around $468\text{ px}$ (IAB banner standard), unlike nonad buttons.* | *80% of nonads have aratio $< 3.0$, while $> 50\%$ of ads exceed $4.0$.* |

---

### 3. Binary Indicator Sparsity & Hierarchical Structure

| Figure 7: Binary Indicator Prevalence (Dot Plot) | Figure 20: Geometry Dendrogram (Lecture 21 Sample) |
|:---:|:---:|
| ![Figure 7](artifacts/report/figures/07_binary_prevalence_dot.png) | ![Figure 20](artifacts/report/figures/20_geometry_dendrogram_sample.png) |
| *Prevalence: `local` ($76.4\%$), `url*images` ($28.8\%$), `origurl*geocities` ($18.5\%$).* | *Average-linkage agglomerative clustering on 120 samples; final merge at distance 4.349.* |

---

## Frequent Itemset Mining & Association Rules (Apriori)

To discover co-occurrence patterns among image attributes, rows were represented as transactions of active binary indicators ($x_j = 1$). Continuous geometry was excluded to avoid arbitrary discretization thresholds. At $s_{\min} = 0.05$ ($164$ transactions) and $c_{\min} = 0.60$, 18 binary indicators qualified for mining.

### Dual-Regime Mining Results

1. **Feature-Only Mining (Unsupervised):**
   - Excludes class labels to discover pure feature co-occurrences.
   - Identified **1,194 frequent itemsets** and **44,122 association rules** in $0.01165\text{ s}$.
2. **Class-Augmented Mining (Supervised Rules):**
   - Appends class tokens (`class=ad`, `class=nonad`).
   - Discovered **2,388 frequent itemsets** and generated **1,190 single-class-consequent rules** ($X \implies \text{class}$) in $0.03741\text{ s}$.
   - **Top Verified Rule:**
     $$\text{\texttt{ancurl*com=1}} \implies \text{\texttt{class=ad}} \quad [s = 5.764\%, c = 91.304\%, \text{lift} = 6.523]$$

<p align="center">
  <img src="artifacts/report/figures/12_apriori_rules_bubble.png" width="70%" alt="Figure 12: Apriori Rules Bubble Plot" />
  <br />
  <em>Figure 12: Support, confidence, and lift distribution across all 1,190 class-consequent association rules.</em>
</p>

### Minimum Support Sensitivity ($k \le 3$)

| Min Support | Min Row Count | Active Items | Frequent Itemsets | Class Rules | Latency |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **2.0%** | 65 | 48 | 3,257 | 531 | 0.0528 s |
| **3.0%** | 98 | 31 | 1,102 | 229 | 0.0175 s |
| **5.0%** | 164 | 18 | 270 | 58 | 0.0051 s |
| **8.0%** | 262 | 10 | 66 | 14 | 0.0019 s |

---

## Closed Frequent Itemsets: The CHARM Algorithm

While Apriori discovers all frequent itemsets, the resulting $44{,}122$ rules exhibit heavy redundancy. We implemented the **CHARM algorithm** (Zaki & Hsiao, 2002) using itemset-tidset (IT-pair) search trees and dynamic equivalence pruning.

An itemset $X$ is a **Closed Frequent Itemset (CFI)** if:
$$X \in \text{CFI} \iff s(X) \ge s_{\min} \quad \text{and} \quad \forall Y \supset X, \; s(Y) < s(X)$$

<p align="center">
  <img src="artifacts/report/figures/13_charm_closed_lengths.png" width="65%" alt="Figure 13: CHARM Closed Itemset Lengths" />
  <br />
  <em>Figure 13: Cardinality distribution of the 41 feature-only Closed Frequent Itemsets discovered by CHARM.</em>
</p>

### Empirical Verification and Compression
- **Feature-Only Space:** CHARM discovered exactly **41 Closed Frequent Itemsets** in $0.000264\text{ s}$ ($264\ \mu\text{s}$).
- **Class-Augmented Space:** Discovered **80 closed itemsets** and **37 non-redundant class rules** in $0.000492\text{ s}$.
- **Exhaustive Mathematical Verification:** Every single closed itemset was independently verified against an exhaustive TID-set intersection test. Zero closure violations and zero missed itemsets were recorded.
- **Compression Efficiency:** CHARM compresses $1{,}194$ frequent patterns into 41 generating closed kernels (**96.57% reduction**), eliminating rule redundancy with zero loss of support information.

---

## Sequential-Pattern Mining: Analysis of Structural Limits

The assignment handout mandates sequential pattern mining. However, adhering to scientific grounding requires stating dataset realities:

### Why Internet Ads Has No Temporal Sequences
1. **Strictly Cross-Sectional Data:** Each row represents a static image extracted from a 1998 web crawl.
2. **No Timestamps or Session IDs:** Consecutive rows are crawler artifacts, not user clickstreams.
3. **Alphabetical Schema Order:** Feature column positions reflect naming conventions, not event chronology. Treating row or column order as time would fabricate artificial findings.

### Dual Experimental Treatment

<p align="center">
  <img src="artifacts/association/schema_role_patterns_top20.png" width="75%" alt="Figure 22: Schema Role Sequence Patterns" />
  <br />
  <em>Figure 22: Top-20 data-derived structural patterns across the analyst-defined hierarchy: $\text{origurl} \to \text{ancurl} \to \text{url}$.</em>
</p>

1. **Synthetic GSP Demonstration ([`toy_sequential_patterns.csv`](artifacts/association/toy_sequential_patterns.csv)):** Generalized Sequential Pattern (GSP) algorithm implemented and verified on five explicit synthetic sequences, producing 14 demonstration patterns.
2. **Data-Derived Structural Role Mining ([`scripts/schema_role_sequential_mining.py`](scripts/schema_role_sequential_mining.py)):** Modeled URL anatomy as an ordered structural sequence:
   $$S_i = \langle (\text{indicators in } \texttt{origurl}) \longrightarrow (\text{indicators in } \texttt{ancurl}) \longrightarrow (\text{indicators in } \texttt{url}) \rangle$$
   Mining all $3{,}279$ rows under this hierarchy yielded **71 frequent structural patterns** and **29 prefix-extension rules**.

> **Explicit Limitation:** These 71 patterns capture cross-field structural co-occurrence, not chronological behavior. True temporal sequence mining remains an unfulfilled requirement of the raw data.

---

## Predictive Classification: Decision Tree (CART)

To classify advertisements while extracting human-interpretable logic, we trained a CART decision tree using scikit-learn.

### Zero-Leakage Grouped Holdout Split
Because 278 feature signatures repeat across $1{,}137$ rows, a standard random train-test split would leak identical feature vectors across partitions. We enforced a **deterministic group-isolated split** (`make_grouped_split`):
- **Train:** $2{,}623$ rows ($80.0\%$), **Test:** $656$ rows ($20.0\%$).
- **Zero group overlap:** Identical feature signatures are strictly constrained to a single partition.
- **Target Distribution:** $13.99\%$ overall, $13.99\%$ train, $14.02\%$ test.
- **Preprocessing Isolation:** Continuous features median-imputed; binary features mode-imputed strictly using training-fold statistics.

| Figure 14: Test Confusion Matrix | Figure 15: Held-Out ROC Curve ($\text{AUC} = 0.9116$) |
|:---:|:---:|
| ![Figure 14](artifacts/report/figures/14_tree_confusion_matrix.png) | ![Figure 15](artifacts/report/figures/15_tree_roc_curve.png) |
| *77 ad hits, 15 missed ads, 38 false alarms ($83.7\%$ recall).* | *Strong ranking capability across decision thresholds ($0.9116$).* |

### Held-Out Performance Metrics ($N = 656$)

| Metric | Nonad (Negative) | Ad (Positive) | Overall / Macro |
|---|:---:|:---:|:---:|
| **Precision** | $0.9723$ | $0.6696$ | $0.8210$ |
| **Recall** | $0.9326$ | $0.8370$ | $0.8848$ |
| **$F_1$-Score** | $0.9520$ | $\mathbf{0.7440}$ | $0.8480$ |
| **Accuracy** | --- | --- | $\mathbf{91.92\%}$ |
| **Balanced Accuracy** | --- | --- | $\mathbf{88.48\%}$ |
| **ROC-AUC** | --- | --- | $\mathbf{0.9116}$ |

### Robustness Check: 15-Fold Repeated Grouped Cross-Validation
To verify that holdout results were not split artifacts, we executed 5-fold GroupKFold across 3 repeats (15 total folds) with train-fold-only imputation ([`scripts/grouped_cv_decision_tree.py`](scripts/grouped_cv_decision_tree.py)):
- **Accuracy:** $0.9328 \pm 0.0128$ [$0.9116 \to 0.9527$]
- **Balanced Accuracy:** $0.8796 \pm 0.0324$ [$0.7890 \to 0.9194$]
- **Ad Precision:** $0.7377 \pm 0.0775$ [$0.6263 \to 0.8621$]
- **Ad Recall:** $0.8057 \pm 0.0646$ [$0.6235 \to 0.8763$]
- **Ad $F_1$-Score:** $\mathbf{0.7676 \pm 0.0562}$ [$0.6463 \to 0.8374$]
- **ROC-AUC:** $0.9002 \pm 0.0256$ [$0.8544 \to 0.9447$]

<p align="center">
  <img src="artifacts/report/figures/16_decision_tree_diagram.png" width="95%" alt="Figure 16: Decision Tree Structure" />
  <br />
  <em>Figure 16: Fitted depth-5 CART decision tree displaying split conditions, Gini impurities, and leaf distributions.</em>
</p>

### Extracted IF-THEN Business Rules
The tree produced 12 terminal leaf rules. Representative verified rules include:
- **Rule 4 (Commercial Banner):**  
  `IF width > 71.0 AND height <= 74.0 AND ancurl*com = 1 THEN class=ad`  
  *(Training rows = 126, weighted leaf ad mass share = 96.8%)*
- **Rule 8 (Editorial Image):**  
  `IF width <= 71.0 AND url*images = 0 AND origurl*geocities = 0 THEN class=nonad`  
  *(Training rows = 1,845, weighted leaf nonad mass share = 94.2%)*

---

## Unsupervised Density Clustering: DBSCAN on Geometry

To determine whether image geometry naturally forms clusters without target labels, we executed DBSCAN on continuous features (`height`, `width`, `aratio`).

### Geometry Scaling & Parameter Justification
1. Continuous features median-imputed, transformed via $\log(1+x)$ to compress heavy tails, and scaled via `RobustScaler` (IQR centering and scaling).
2. Set $\text{MinPts} = 10$ ($> 2 \times \text{dim} = 6$).
3. Plotted the sorted 10th-nearest-neighbor distance curve across all $3{,}279$ points (Figure 17). The maximum perpendicular distance from the endpoint chord identified the knee at $\mathbf{\varepsilon = 0.457798}$.

| Figure 17: Sorted 10-NN Distance Curve (Elbow Knee) | Figure 18: Spatial Clusters & Noise in Geometry Space |
|:---:|:---:|
| ![Figure 17](artifacts/report/figures/17_dbscan_k_distance.png) | ![Figure 18](artifacts/report/figures/18_dbscan_clusters.png) |
| *Chord-distance knee heuristic selects $\varepsilon = 0.4578$ for $\text{MinPts}=10$.* | *Partitions data into 3 dense clusters and 70 noise points ($2.13\%$).* |

### Clustering Results & Post-Hoc Label Independence
- **Discovered Structure:** **3 dense spatial clusters** and **70 noise instances** ($2.13\%$) with a non-noise silhouette score of **0.4991**.
- **Class Alignment Check:** The post-hoc Adjusted Rand Index (ARI) against commercial ad labels is $\mathbf{-0.0240}$ (and $-0.0176$ excluding noise).
- **Inference:** **Unsupervised geometric density does not separate commercial advertisements from editorial images.** Commercial ads span diverse form factors (square buttons, skyscraper banners, header leaderboards), overlapping heavily with organic web graphics.

---

## PySpark Distributed Analytics vs. Local Benchmarking

We implemented matched machine-learning pipelines in Apache Spark 3.5.3 (Java 11, PySpark MLlib) in `local[2]` mode on an Intel Core i7-13700HX (24 cores visible, Linux).

### Real-Data Benchmark on 3,279 Rows

<p align="center">
  <img src="artifacts/report/figures/19_spark_runtime_comparison.png" width="70%" alt="Figure 19: Spark vs Local Runtime Comparison" />
  <br />
  <em>Figure 19: Log-scale runtime comparison between local implementations and PySpark 3.5.3 on 3,279 rows.</em>
</p>

| Workload | Local Implementation | PySpark MLlib (`local[2]`) | Slowdown Ratio | Accuracy / Result |
|---|:---:|:---:|:---:|:---:|
| **Decision Tree Fit** | **0.0306 s** (Cython CART) | **0.8652 s** (Spark MLlib) | **Spark 28.2× slower** | $91.92\%$ local vs. $92.07\%$ Spark |
| **Decision Tree End-to-End** | **0.0323 s** (Fit + Predict) | **5.7000 s** (Conv + Fit + Collect) | **Spark 176.2× slower** | Identical test split & preprocessing |
| **Frequent Pattern Mining** | **0.0380 s** (Local Apriori) | **0.0712 s** (Spark FP-Growth) | **Spark 1.9× slower** | Both found 2,388 itemsets |
| **Pattern Mining End-to-End** | **0.0380 s** (Mining) | **0.4459 s** (Conv + Fit + Collect) | **Spark 11.7× slower** | Matching 11,709 singleton rules |
| *Spark JVM Startup* | *0.0000 s* | *1.5253 s (measured separately)* | --- | Baseline Py4J gateway initialization |

---

### Synthetic Scaling Sweep ($3.2\text{k} \to 500\text{k}$ Rows)

To investigate whether a performance crossover occurs as data scales, we executed [`scripts/scaled_tree_spark_benchmark.py`](scripts/scaled_tree_spark_benchmark.py), expanding training data from $3{,}279$ to $500{,}000$ rows.

<p align="center">
  <img src="artifacts/spark/scaled_tree_benchmark.png" width="75%" alt="Figure 23: Synthetic Scaling Benchmark Curves" />
  <br />
  <em>Figure 23: Log-log scaling curves comparing local Cython CART fit vs. PySpark MLlib fit across 3,279 to 500,000 rows.</em>
</p>

| Synthetic Rows | Local CART Fit | Spark CART Fit | Spark Total Task Time | Fit Ratio | E2E Ratio |
|---:|:---:|:---:|:---:|:---:|:---:|
| **3,279** (base) | **0.0075 s** | 1.8755 s | 4.0438 s | Spark $249.7\times$ slower | Spark $513.3\times$ slower |
| **10,000** | **0.0141 s** | 0.7982 s | 1.2772 s | Spark $56.7\times$ slower | Spark $87.1\times$ slower |
| **25,000** | **0.0305 s** | 0.8192 s | 1.6951 s | Spark $26.8\times$ slower | Spark $54.1\times$ slower |
| **50,000** | **0.0622 s** | 1.0642 s | 2.6384 s | Spark $17.1\times$ slower | Spark $41.7\times$ slower |
| **100,000** | **0.1436 s** | 2.6942 s | 5.7900 s | Spark $18.8\times$ slower | Spark $39.8\times$ slower |
| **250,000** | **0.4229 s** | 5.7544 s | 13.2393 s | Spark $13.6\times$ slower | Spark $30.9\times$ slower |
| **500,000** | **0.9313 s** | 11.0962 s | 25.3173 s | **Spark 11.9× slower** | **Spark 26.8× slower** |

### Why Single-Node Spark Does Not Exhibit a Crossover
1. **Scikit-Learn Sparse Optimization:** Scikit-learn’s tree engine is compiled Cython operating directly on contiguous CSR sparse matrix pointers in RAM. At depth 5 with $1\%$ sparsity, evaluating 31 split thresholds on $500{,}000$ rows takes **under 1 second** ($0.93\text{ s}$).
2. **Distributed Overhead in Local Mode:** Spark MLlib in `local[2]` incurs Py4J socket serialization, DataFrame conversion, Tungsten serialization, multi-threaded task scheduling, and partition histogram synchronization.
3. **1,000,000 Row Memory Boundary:** Pushing the benchmark to $1{,}000{,}000$ rows caused Spark's DataFrame cache to fail with:
   ```text
   java.lang.OutOfMemoryError: Java heap space
       at java.base/java.nio.ByteBuffer.allocate(ByteBuffer.java:348)
       at org.apache.spark.sql.execution.columnar.ColumnBuilder$.ensureFreeSpace(ColumnBuilder.scala:167)
   ```
   In contrast, local sparse CSR arrays comfortably fit in available host RAM.
4. **Architectural Takeaway:** Spark's true speedup requires **multi-node cluster distribution** across networked nodes processing terabyte-scale data exceeding single-machine RAM. On a single workstation where data fits in memory, local Cython execution is consistently superior.

---

## Experiment Provenance with MLflow

All model runs, preprocessing configurations, and benchmark outputs were tracked in an embedded SQLite MLflow repository ([`artifacts/mlflow.db`](artifacts/mlflow.db)) under Experiment ID 2 (`Internet Ads Hackathon (final artifacts)`):

- **Logged Records:** 8 runs, 50 parameters, 83 metrics, and 25 artifacts across 4 main modules.
- **Decision Tree Run (`bef48765...`):** Tracks depth 5, balanced weights, test accuracy ($0.9192$), ad-$F_1$ ($0.7440$), ROC-AUC ($0.9116$), and serialized IF-THEN rules.
- **DBSCAN Run (`452a0bdf...`):** Tracks $\varepsilon=0.4578$, $\text{MinPts}=10$, 3 clusters, 70 noise rows, and silhouette score ($0.4991$).
- **PySpark Benchmark Run (`26edb63a...`):** Tracks startup latency ($1.525\text{ s}$), conversion times, local and Spark fit/predict timings, and slowdown ratios.

---

## Instructions to Run & Reproduce

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/ArpitKrSingh7/bdHackathon.git
cd bdHackathon

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Execute Data Quality Audit
```bash
python3 scripts/data_quality_audit.py
```

### 3. Run Association & CHARM Mining
```bash
python3 association_mining.py
```

### 4. Execute Decision Tree, Grouped CV & DBSCAN
```bash
# Baseline CART and DBSCAN
python3 classification_clustering.py

# 15-Fold Repeated Grouped Cross-Validation
python3 scripts/grouped_cv_decision_tree.py
```

### 5. Run PySpark Benchmarks & Scaling Sweep
```bash
# Matched 3,279-row benchmark
python3 pyspark_benchmark.py

# Synthetic scaling sweep (3.2k to 500k rows)
python3 scripts/scaled_tree_spark_benchmark.py --sizes 3279 10000 25000 50000 100000 250000 500000 --repeats 1
```

### 6. Generate Figures & Log MLflow
```bash
python3 report_figures.py
python3 log_experiments.py
```

### 7. Compile LaTeX Report on Overleaf
1. Upload the [`report_latex/`](report_latex/) directory to [Overleaf](https://www.overleaf.com/).
2. Set `main.tex` as the root document and compile using **pdfLaTeX**.

---

## References

1. Kushmerick, N. (1999). *Internet Advertisements* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C5V011](https://doi.org/10.24432/C5V011).
2. Agrawal, R., Imieliński, T., & Swami, A. (1993). "Mining Association Rules between Sets of Items in Large Databases." *ACM SIGMOD '93*, 207–216.
3. Zaki, M. J., & Hsiao, C.-J. (2002). "CHARM: An Efficient Algorithm for Closed Itemset Mining." *SIAM SDM 2002*, 457–473.
4. Srikant, R., & Agrawal, R. (1996). "Mining Sequential Patterns: Generalizations and Performance Improvements." *EDBT 1996*, 3–17.
5. Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise." *KDD-96*, 226–231.
6. Apache Spark. (2024). *Spark 3.5.3 MLlib Documentation*. [https://spark.apache.org/docs/3.5.3/ml-guide.html](https://spark.apache.org/docs/3.5.3/ml-guide.html).
7. Course Slides: *25CS5004 / CS 5010 Analytics & Systems of Big Data Practice*, Dr. B. Sivaselvan, IIITDM Kancheepuram (Lectures 4, 6, 9, 11, 13, 20, 21).

---

## Academic Integrity & Verification Guarantee

*This project was developed under strict adherence to scientific grounding. No metrics, rule counts, execution latencies, or plot interpretations have been fabricated. All source code, data audit logs, model outputs, and LaTeX sources are completely verified and self-contained.*
