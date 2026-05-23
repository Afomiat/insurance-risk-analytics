The issue is that GitHub README needs proper markdown formatting. Let me give you the complete file to copy-paste directly into your README.md in VS Code.

---

Open `README.md` in VS Code, select ALL the content (`Ctrl+A`) and delete it, then paste this entire block:

```markdown
# 🛡️ Insurance Risk Analytics & Predictive Modeling

### AlphaCare Insurance Solutions (ACIS) — South Africa

![Python](https://img.shields.io/badge/Python-3.10-blue)
![DVC](https://img.shields.io/badge/DVC-3.x-purple)
![CI](https://img.shields.io/badge/CI-GitHub_Actions-green)
![Status](https://img.shields.io/badge/Status-In_Progress-orange)

---

## 📋 Project Overview

| Field | Detail |
|---|---|
| **Company** | AlphaCare Insurance Solutions (ACIS) |
| **Role** | Marketing Analytics Engineer |
| **Dataset** | 18 months of South African car insurance data |
| **Period** | February 2014 – August 2015 |
| **Records** | 1,000,098 policies |

ACIS is preparing for aggressive growth in the South African auto-insurance market. This project delivers evidence-driven strategies to optimize marketing investments and refine pricing models — moving beyond intuition-based pricing toward analytics-driven decisions grounded in historical claim data, statistical rigor, and machine learning.

---

## 🎯 Business Objectives

1. Build a deep understanding of insurance risk metrics
2. Statistically validate key hypotheses about risk drivers across provinces, zip codes, and gender
3. Develop predictive models that estimate claim severity and probability of a claim
4. Communicate findings in a clear business-facing report that ACIS leadership can act on

---

## 📊 Core Business Metrics

Two derived metrics anchor the entire analysis:

| Metric | Formula | Meaning |
|---|---|---|
| **Loss Ratio** | TotalClaims ÷ TotalPremium | Greater than 1.0 means losing money |
| **Margin** | TotalPremium − TotalClaims | Per-policy profit in Rand |

---

## 🗂️ Project Structure

```
insurance-risk-analytics/
│
├── .github/
│   └── workflows/
│       └── ci.yml                           # GitHub Actions CI pipeline
│
├── data/                                    # Tracked by DVC, NOT Git
│   ├── MachineLearningRating_v3.txt         # v1: raw data
│   ├── MachineLearningRating_v3.txt.dvc     # DVC pointer file
│   ├── cleaned_insurance_data.csv           # v2: cleaned data
│   └── cleaned_insurance_data.csv.dvc       # DVC pointer file
│
├── notebooks/
│   ├── 01_eda.ipynb                         # Exploratory Data Analysis
│   ├── 02_hypothesis_testing.ipynb          # Coming in Task 3
│   └── 03_modeling.ipynb                    # Coming in Task 4
│
├── src/
│   ├── __init__.py                          # Python package marker
│   ├── data_loader.py                       # Load, fix types, add metrics
│   ├── data_cleaner.py                      # Full cleaning pipeline
│   ├── eda_utils.py                         # Reusable EDA functions
│   ├── hypothesis_tests.py                  # Coming in Task 3
│   └── modeling.py                          # Coming in Task 4
│
├── reports/
│   ├── final_report.md                      # Final business report
│   ├── missing_values.png                   # Data quality chart
│   ├── financial_distributions.png          # Distribution analysis
│   ├── loss_ratio_by_province.png           # Regional risk chart
│   ├── outlier_boxplots.png                 # Outlier detection
│   ├── temporal_trends.png                  # 18-month trends
│   ├── claims_by_vehicle_make.png           # Vehicle risk profiling
│   ├── premium_vs_claims_zipcode.png        # Zip code scatter
│   ├── correlation_matrix.png               # Feature correlations
│   ├── geographic_trends.png               # Province analysis
│   └── loss_ratio_vehicle_gender.png        # Demographic risk
│
├── tests/                                   # Automated tests
├── .dvc/                                    # DVC configuration
├── .gitignore                               # Git ignore rules
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git installed
- pip package manager

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/insurance-risk-analytics.git
cd insurance-risk-analytics
```

### Step 2 — Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Pull the Data via DVC

```bash
# Configure remote storage (first time only)
dvc remote add -d localstorage /path/to/your/dvc-storage

# Pull both data versions
dvc pull
```

### Step 5 — Launch Notebooks

```bash
jupyter notebook
```

Open `notebooks/01_eda.ipynb` and select the venv kernel.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| pandas | >= 1.5.0 | Data manipulation |
| numpy | >= 1.23.0 | Numerical computing |
| matplotlib | >= 3.6.0 | Plotting |
| seaborn | >= 0.12.0 | Statistical visualization |
| scikit-learn | >= 1.2.0 | Machine learning |
| xgboost | >= 1.7.0 | Gradient boosting |
| shap | >= 0.41.0 | Model interpretability |
| jupyter | latest | Notebook environment |
| dvc | latest | Data version control |
| flake8 | latest | Code linting |
| pytest | latest | Automated testing |

---

## ✅ Task 1 — Git, GitHub & Exploratory Data Analysis

### What Was Asked

- Set up a reproducible development environment with GitHub
- Configure GitHub Actions CI pipeline for linting and testing
- Perform EDA covering 6 analysis areas
- Answer 4 guiding business questions
- Produce at least 3 creative visualizations

### What We Built

#### Environment Setup

- GitHub repository with Python `.gitignore` and `requirements.txt`
- Virtual environment (`venv`) isolating all project packages
- Modular project structure separating notebooks from reusable source code
- Branch `task-1` with minimum 3 commits per day using Conventional Commits format

#### GitHub Actions CI Pipeline

Every push to any branch automatically triggers:

1. Spins up a fresh Ubuntu Linux environment
2. Installs Python 3.10
3. Installs all project dependencies
4. Runs `flake8` linting across `src/`
5. Runs `pytest` across `tests/`

This ensures code quality is enforced at every step and no broken code reaches `main`.

#### Source Modules Built

**`src/data_loader.py`**

| Function | What It Does |
|---|---|
| `load_data()` | Detects file type, handles pipe-delimited format |
| `fix_dtypes()` | Converts dates, ensures numerical columns are numeric |
| `standardize_columns()` | Fixes mixed-case names in raw data |
| `add_derived_metrics()` | Adds LossRatio and Margin columns |
| `load_and_prepare()` | Master function — one call does everything |

> **Key discovery:** Raw data had inconsistent column casing — `make`, `kilowatts`, `mmcode`, `cubiccapacity`, `bodytype` were all lowercase while other columns were Title Case. Fixed in `standardize_columns()` so all downstream code uses consistent names.

### EDA Findings

#### Data Summarization

- Dataset shape: **1,000,098 rows × 53 columns**
- Column groups: Policy, Client, Location, Vehicle, Plan, Financial
- `TransactionMonth` correctly parsed as `datetime64`
- Numerical columns confirmed as `float64` or `int64`

#### Data Quality Assessment

Multiple columns had missing values. Strategy applied:

| Rule | Action | Reason |
|---|---|---|
| Column > 50% missing | Drop column | Too little data to be useful |
| Numerical < 50% missing | Fill with median | Median is robust to skew |
| Categorical < 50% missing | Fill with mode | Most frequent = best guess |
| TotalClaims missing | Leave as-is | Missing means no claim occurred |

#### Univariate Analysis

- `TotalClaims` is heavily **right-skewed** — most policies have zero or small claims, a few are catastrophically large
- Log transformation of TotalClaims produces a near-normal distribution — critical for modeling in Task 4
- `TotalPremium` is more normally distributed with some outliers

#### Bivariate and Multivariate Analysis

- Scatter plot of TotalPremium vs TotalClaims colored by PostalCode reveals clear geographic clustering of risk
- Points above the break-even line (where Claims > Premium) show unprofitable policies concentrated in specific zip codes
- Correlation matrix shows strongest relationships between SumInsured, CalculatedPremiumPerTerm, and TotalPremium

#### Geographic Trends

- Premium levels vary significantly across South African provinces
- Toyota is the dominant vehicle make across most provinces
- Cover type preferences and vehicle choices differ meaningfully by region

#### Outlier Detection

- Box plots on TotalPremium, TotalClaims, and CustomValueEstimate reveal significant extreme values
- Outliers handled in the cleaning pipeline using a conservative 3× IQR threshold to preserve legitimate high-value claims

### Guiding Questions Answered

**Q1: What is the overall Loss Ratio? How does it vary?**
- Overall portfolio Loss Ratio calculated from sum of all claims divided by sum of all premiums
- Broken down by Province, VehicleType, and Gender using grouped aggregation
- Provinces colored red (above average risk) vs green (below average) for immediate visual clarity

**Q2: Distributions of key financial variables — outliers?**
- TotalClaims is right-skewed with significant outliers at the high end
- CustomValueEstimate also has extreme outliers representing high-value vehicles
- Both handled via IQR method in the cleaning pipeline

**Q3: Temporal trends over 18 months?**
- Average premium, average claims, and claim frequency tracked monthly
- Policy count per month shows portfolio growth trend
- Any seasonal spikes in claim frequency identified

**Q4: Which vehicle makes have highest and lowest claims?**
- Minimum 100 policies required per make for statistical reliability
- Top 10 highest and lowest average claim makes identified
- Direct input into risk-based premium adjustments

### Visualizations Produced

| Plot | File | What It Shows |
|---|---|---|
| Missing values bar chart | `missing_values.png` | Which columns have data gaps |
| Financial distributions | `financial_distributions.png` | Skew and shape of key variables |
| Loss ratio by province | `loss_ratio_by_province.png` | Regional profitability map |
| Outlier box plots | `outlier_boxplots.png` | Extreme value detection |
| Temporal trends | `temporal_trends.png` | 18-month claim patterns |
| Vehicle make claims | `claims_by_vehicle_make.png` | Make-level risk ranking |
| Premium vs claims scatter | `premium_vs_claims_zipcode.png` | Zip code risk clustering |
| Correlation heatmap | `correlation_matrix.png` | Feature relationships |
| Geographic 3-panel | `geographic_trends.png` | Province behavior overview |
| Loss ratio by vehicle/gender | `loss_ratio_vehicle_gender.png` | Demographic risk breakdown |

### Task 1 Deliverables

```
✅ GitHub repository with working CI pipeline
✅ task-1 branch with minimum 3 commits per day
✅ EDA notebook (notebooks/01_eda.ipynb)
✅ Reusable source modules in src/
✅ 10 visualizations produced (minimum required was 3)
✅ All 4 guiding questions answered with evidence
✅ Merged into main via Pull Request
```

---

## 🗄️ Task 2 — Data Version Control (DVC)

### What Was Asked

- Merge task-1 into main via Pull Request
- Install and initialize DVC
- Set up local remote storage outside the project
- Track the raw dataset as version 1
- Create a cleaned version as version 2
- Push both versions to remote storage
- Document the reproduction pipeline

### Why DVC in Insurance?

In regulated industries like insurance, every model result must be reproducible. Regulators can demand at any time:

> *"Show us exactly what data you used to set premiums in Month X"*

| Without DVC | With DVC |
|---|---|
| Difficult to reconstruct exact data | One command restores exact data |
| No audit trail for data changes | Full version history |
| Manual file management | Automated pipeline |
| Risk of compliance failure | Audit-ready at all times |

```bash
# Restore exact data from any past commit
git checkout <commit-hash>
dvc pull
```

### How DVC Works Alongside Git

```
Git  tracks  → code files (notebooks, scripts, configs)
DVC  tracks  → data files (CSVs, models, large files)

What Git sees:
  data/MachineLearningRating_v3.txt.dvc   ← tiny 200-byte pointer

What DVC stores in remote:
  The actual 500MB data file, identified by its MD5 hash
```

The MD5 hash in the `.dvc` file is a unique fingerprint. If even one byte of data changes, the hash changes. This guarantees you always know exactly which data was used.

### Data Versions Created

| Version | File | Rows | Columns | Description |
|---|---|---|---|---|
| **v1 raw** | `MachineLearningRating_v3.txt` | 1,000,098 | 53 | Original data as received |
| **v2 clean** | `cleaned_insurance_data.csv` | 859,867 | 48 | After full cleaning pipeline |

**140,231 rows removed (14.0%) during cleaning:**

| Step | Action | Rows Affected |
|---|---|---|
| Deduplication | Removed identical records | Variable |
| Column dropping | Dropped columns > 50% missing | 5 columns removed |
| Missing imputation | Filled remaining gaps | 0 rows removed |
| Outlier removal | 3× IQR on 3 key columns | Remaining removals |

### Cleaning Pipeline

**`src/data_cleaner.py`** runs three sequential steps:

```
Step 1: remove_duplicates()
        Identical rows are data entry errors in insurance.
        Removed before any analysis.

Step 2: handle_missing_values()
        Columns > 50% missing   → dropped entirely
        Numerical remaining     → filled with median
        Categorical remaining   → filled with mode
        TotalClaims missing     → left as-is (no claim = no value)

Step 3: remove_outliers()
        Method: IQR with 3.0 multiplier (conservative)
        Columns: TotalPremium, TotalClaims, CustomValueEstimate
        Why 3.0 not 1.5: preserves legitimate high-value claims
```

### Reproduce the Data Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Configure DVC remote (first time only)
dvc remote add -d localstorage /path/to/your/storage

# Pull both data versions from remote
dvc pull

# Or regenerate cleaned data from raw
python src/data_cleaner.py
```

### DVC Remote Storage Structure

```
dvc-storage/
└── insurance-risk-analytics/
    └── files/
        └── md5/
            ├── 9b/    ← v1 raw data
            └── f6/    ← v2 cleaned data
```

### Task 2 Deliverables

```
✅ task-1 merged into main via PR
✅ task-2 branch created from updated main
✅ DVC installed and initialized (dvc init)
✅ Local remote storage configured
✅ Raw data v1 tracked — 1,000,098 rows
✅ Cleaned data v2 tracked — 859,867 rows
✅ Both versions pushed to remote storage
✅ .dvc pointer files committed to Git
✅ src/data_cleaner.py built and committed
✅ README updated with reproduction steps
✅ task-2 merged into main via PR
```

---

## 🔬 Task 3 — A/B Hypothesis Testing

### Status: In Progress

Four null hypotheses will be statistically tested to form the evidence base for ACIS's new segmentation and pricing strategy:

| Hypothesis | Null Statement | Test | KPI |
|---|---|---|---|
| H1 | No risk differences across provinces | z-test | Claim Frequency |
| H2 | No risk differences between zip codes | z-test | Claim Severity |
| H3 | No margin difference between zip codes | z-test | Margin |
| H4 | No risk difference between genders | z-test | Claim Frequency |

**Decision rule:** Reject H₀ when p-value < 0.05

---

## 🤖 Task 4 — Statistical Modeling & Risk-Based Pricing

### Status: Coming Soon

Three models will be built, tuned, and compared:

| Model | Type | Use Case |
|---|---|---|
| Linear Regression | Baseline | Claim severity prediction |
| Random Forest | Ensemble | Non-linear risk patterns |
| XGBoost | Gradient Boost | Best predictive performance |

**Target:** TotalClaims (claim severity)  
**Evaluation:** RMSE and R²  
**Interpretability:** SHAP feature importance plots

---

## 📅 Key Dates

| Milestone | Date |
|---|---|
| Challenge Introduction | 20 May 2026 |
| Interim Submission | 24 May 2026 |
| Final Submission | 26 May 2026 |

---

## 👥 Team

| Role | Names |
|---|---|
| Tutors | Kerod, Mahbubah, Feven |

---

## 📄 License

MIT License — see LICENSE file for details.

---

*Last updated: May 2026*
```

---

