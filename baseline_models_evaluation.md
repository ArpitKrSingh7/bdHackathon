# Baseline Models Evaluation & Compliance Audit

**Target Problem:** Analytics & Systems of Big Data Practice (CS 5010 / 25CS5004) Endsem  
**Dataset:** [UCI Internet Advertisements](https://archive.ics.uci.edu/dataset/51/internet+advertisements)  
**Raw Source Files:** `data/raw/ad.data` (3,279 rows × 1,559 columns), `data/raw/ad.names`, `data/raw/ad.DOCUMENTATION`  
**Reference Folder:** `someBaseLineModels(otherWorks)/`  
**Course Slides:** `../endSem/` (Lectures 2–22)  
**Audit Script:** Reproducible via [`scripts/audit_dataset.py`](scripts/audit_dataset.py)

---

## 1. Verified Raw Dataset Facts (Direct Script Execution)

The following statistics were computed directly from `data/raw/ad.data` using [`scripts/audit_dataset.py`](scripts/audit_dataset.py) under Python 3.10.12, pandas 2.3.2, and SciPy 1.15.3:

* **Dimensions:** 3,279 rows × 1,559 columns (1,558 input features declared in `ad.names` + 1 target class label).
* **Target Class Distribution (`class`):**
  * `nonad`: **2,820** instances (86.00%)
  * `ad`: **459** instances (14.00%)
  * *Discrepancy Note:* UCI's `ad.DOCUMENTATION` states 2,821 nonad and 458 ad. The actual parsed `ad.data` file contains 2,820 and 459. All downstream modeling and evaluation must use the actual file counts.
* **Missing Values by Feature:**
  * Column 0 (`height`): **903** missing (27.54%)
  * Column 1 (`width`): **901** missing (27.48%)
  * Column 2 (`aratio`): **910** missing (27.75%)
  * Column 3 (`local`): **15** missing (0.46%)
  * Columns 4 through 1,557: **0** missing
* **Row-Level Missingness:**
  * Rows missing at least one geometry feature (`height`, `width`, `aratio`): **910 rows** (27.75%)
  * Rows missing at least one input feature (including `local`): **920 rows** (28.06%)
* **Moment Skewness (Geometry Features):**
  * `height`: **2.47** (pandas sample skewness: 2.47)
  * `width`: **1.62** (pandas sample skewness: 1.63)
  * `aratio`: **7.03** (pandas sample skewness: 7.03)
  * *Correction:* Only `aratio` has skewness $> 5$; `height` and `width` exhibit moderate-to-high positive skewness (~2.47 and ~1.62).

---

## 2. Baseline Notebooks Compliance Matrix

The 4 baseline notebooks in `someBaseLineModels(otherWorks)/` were inspected cell-by-cell against the assignment requirements:

| Requirement | Exam Sheet Specification | Notebook 1: `decision-trees-vs-logistic-regression` | Notebook 2: `internet-advertisements-detector-optimised` | Notebook 3: `sales-prediction-using-python` | Notebook 4: `web-sayfalar-ndaki-reklam-resimlerinin-tespiti` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Dataset Match** | UCI Internet Advertisements | ⚠️ Partial (`add.csv`, incomplete) | ✅ Yes (`add.csv`) | ❌ **No** (`sales.csv` TV/Radio/Sales) | ✅ Yes (`add.csv`) |
| **Data Cleaning** | Handle missing values (`?` in geometry & local) | ⚠️ Minimal `fillna(-1)` imputation; unexecuted and not justified/validated | ⚠️ Replaces missing values using column mean | ❌ N/A (Wrong dataset) | ⚠️ Replaces missing values using column mean |
| **Data Selection** | Identify task-relevant attributes (reduce 1558 cols) | ⚠️ `df.iloc[:, 1:-1]` drops the first feature; reason is unstated | ❌ None (used raw features) | ❌ N/A | ⚠️ Computed PCA component variances |
| **Data Transformation**| Normalization & discretization | ⚠️ Missing values represented by `-1`; no documented scaling/discretization | ⚠️ StandardScaler only | ❌ N/A | ⚠️ StandardScaler only |
| **10 Course Plots** | 10 distinct plots from course slides | ❌ None | ❌ 3–4 standard plots | ❌ N/A | ❌ 4–5 plots (in Turkish) |
| **Predictive 1: Apriori**| Apriori Sequential Patterns Version | ❌ **Missing** | ❌ **Missing** | ❌ **Missing** | ❌ **Missing** |
| **Predictive 2: DT / DBSCAN**| Decision Tree (with explicit if-then rules) / DBSCAN | ⚠️ Unexecuted classifier code (no rules) | ❌ Used KMeans only | ❌ N/A | ⚠️ CART in CV loop only (no rules/DBSCAN) |
| **Predictive 3: CHARM** | Closed Frequent Itemsets & Rules via CHARM | ❌ **Missing** | ❌ **Missing** | ❌ **Missing** | ❌ **Missing** |
| **Big Data Platform** | PySpark benchmark for 2 techniques + speedup analysis | ❌ **Missing** | ❌ **Missing** | ❌ **Missing** | ❌ **Missing** |
| **Deliverable PDF** | Comprehensive report with interpretations | ❌ None | ❌ None | ❌ None | ❌ None |
| **FINAL VERDICT** | **QUALIFIES FOR SUBMISSION?** | ❌ **DOES NOT QUALIFY** | ❌ **DOES NOT QUALIFY** | ❌ **DOES NOT QUALIFY** | ❌ **DOES NOT QUALIFY** |

---

## 3. Cell-by-Cell Audit of Baseline Notebooks

### A. Notebook 1: `decision-trees-vs-logistic-regression.ipynb`
* **Contents:** 7 code cells. Defines `train_test_split`, `DecisionTreeClassifier`, `LogisticRegression`, and `classification_report` on `add.csv`.
* **Evaluation:** An unexecuted minimal classifier draft. The cells contain code to train standard scikit-learn models, but no cells have been executed, no outputs or metrics are saved, and there is no logic for extracting human-readable IF-THEN decision rules.
* **Verdict:** Does not qualify. Lacks preprocessing, visualization, rule extraction, ARM, clustering, and PySpark.

### B. Notebook 2: `internet-advertisements-detector-optimised.ipynb`
* **Contents:** 85 cells. Replaces missing values in geometry columns with the column mean. Evaluates `GaussianNB`, `SVC`, and `KNeighborsClassifier`. Applies $K$-Means clustering with an elbow curve.
* **Evaluation:**
  1. *Mean Imputation Risk:* Global mean imputation is heavily distorted by the positive skewness in `height` and `width`.
  2. *Algorithm Mismatch:* The assignment asks for Decision Tree classification / DBSCAN clustering; this notebook used SVC, KNN, and $K$-Means.
  3. *Missing Components:* Zero ARM (Apriori), zero CHARM, zero PySpark.
* **Verdict:** Does not qualify. Useful primarily for demonstrating regex parsing of `'?'` with variable leading whitespace.

### C. Notebook 3: `sales-prediction-using-python.ipynb`
* **Contents:** 29 cells using `/kaggle/input/sales-marketing/sales.csv` (`TV`, `Radio`, `Newspaper`, `Sales`).
* **Verdict:** Does not qualify. Completely unrelated regression dataset with zero relevance to the Internet Ads classification task.

### D. Notebook 4: `web-sayfalar-ndaki-reklam-resimlerinin-tespiti.ipynb`
* **Contents:** 88 cells written in Turkish. Examines missing data, runs 10-fold CV on standard classifiers, and fits a PCA model.
* **Evaluation of PCA Code:**  
  Cell 76 runs `pca = PCA(); pca.fit(X); plt.plot(pca.explained_variance_)`. This plots **individual eigenvalues** ($\lambda_i$)—that is, the variance captured by each orthogonal principal component. It does NOT plot the variance of the original raw binary features, nor does it compute cumulative explained variance ratio (`np.cumsum(pca.explained_variance_ratio_)`). Consequently, any claim that this notebook demonstrated ">90% cumulative variance in 200 components" is unsupported by its code.
* **Verdict:** Does not qualify. Missing Apriori, CHARM, DBSCAN, decision tree rule extraction, and PySpark.

---

## 4. Methodological Guidelines & Scope Caveats

### A. Preprocessing & Imputation Standards (Avoiding Data Leakage)
* **Lecture 5 (KDD Phases, slide 8):** Conceptualizes missing data handling (ignore tuple, global constant, attribute mean, class-specific mean, inference-based).
* **Predictive Pipeline Constraint:** Using the target class label $y$ to impute feature values $X$ leaks label information into the feature representations. In real-world inference and proper supervised evaluation, the test label is unavailable. Therefore:
  * For **predictive pipelines**, use **feature-only imputation** (e.g. median or mean computed strictly from $X_{\text{train}}$ without reference to $y_{\text{train}}$).
  * Class-conditional statistics should be reserved strictly for **exploratory/descriptive data analysis**.

### B. Assignment Wording Ambiguity ("Decision Tree classification / DBSCAN clustering")
* The assignment states: *"Apply Decision Tree classification / DBSCAN clustering algorithm that is relevant for the data set provided"*.
* The slash (`/`) is linguistically ambiguous: it may mean choosing one technique (supervised classification vs. unsupervised clustering based on relevance), or it may mean demonstrating both.
* Rather than assuming the instructor's grammatical intent, the safest strategy for maximizing marks is to **implement both** if time permits:
  1. Train a Decision Tree classifier, evaluate performance metrics, and extract explicit IF-THEN rules.
  2. Fit DBSCAN clustering on the normalized geometry attributes and report cluster labels, noise points, and clustering quality.

### C. Sequential Pattern Mining on Cross-Sectional Data
* The Internet Advertisements dataset is static, cross-sectional tabular data consisting of image geometry and page/URL token indicators. It contains no timestamps, session identifiers, or natural event sequences.
* Applying sequential pattern mining (e.g. GSP or AprioriAll) requires imposing an artificial ordering (e.g. ordering features alphabetically or by schema prefix: URL $\to$ Anchor $\to$ Geometry).
* **Documentation Rule:** Any such sequence formulation must be explicitly documented as an **artificial algorithmic demonstration** created solely to fulfill the exam prompt. Its outputs must NOT be interpreted as real temporal user behavior, and ordinary Apriori itemset mining should be presented as the primary association analysis.

### D. Course Slide Alignment for CHARM and DBSCAN
* The course slides cover **A-Close** (Lecture 11) for closed itemsets, **Pincer Search** (Lecture 12) for maximal itemsets, and **$K$-Means / Hierarchical Clustering** (Lectures 20 & 21).
* **CHARM and DBSCAN do not appear in the provided slide PDFs.** We implement CHARM (IT-tree, tidset intersections, closure properties 1–4) and DBSCAN because the **assignment prompt explicitly demands them**, while acknowledging that they were not part of the provided lecture slide deck.

### E. Course-Taught Visualizations
* To satisfy the "at least 10 unique plots discussed in the course" requirement, we prioritize forms taught in the slides:
  1. **Lecture 4:** Pie Chart, Histogram, Bar Graph, Dot Plot, Stem-and-Leaf Plot, Scatter Plot, Box Plot.
  2. **Lecture 13:** ROC Curve (TPR vs. FPR, AUC).
  3. **Lectures 20 & 21:** Dendrogram (hierarchical clustering tree).
  4. Diagnostic plots (such as confusion matrices or correlation matrices) should be clearly labeled as standard machine learning diagnostic tools rather than slide-taught visualizations.

---

## 5. Audit Script Execution

To reproduce all dataset dimensions, class distributions, missingness statistics, and moment skewness figures reported in this document, run:

```bash
python3 scripts/audit_dataset.py
```
*(Environment: Python 3.10.12, pandas 2.3.2, SciPy 1.15.3)*
