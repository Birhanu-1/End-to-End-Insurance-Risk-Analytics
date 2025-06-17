# End-to-End-Insurance-Risk-Analytics--and---Predictive-Modeling
# 🚗 AlphaCare Insurance Solutions – Car Insurance Analytics

Welcome to the Week 1 project of the Marketing Analytics initiative at AlphaCare Insurance Solutions (ACIS). The goal of this project is to analyze historical insurance claim data from South Africa to identify low-risk client segments and optimize marketing strategies.

## 📁 Project Overview

**Objective:**  
- Analyze car insurance data to uncover risk profiles and profitability patterns.
- Develop insights for marketing strategy and premium adjustment.
- Identify low-risk targets and trends in claims data.

## 📌 Tasks Completed (Task 1)

1. **GitHub Setup**
   - Initialized Git repo with CI/CD workflows.
   - Created separate branch `task-1`.

2. **Exploratory Data Analysis (EDA)**
   - Data summary and profiling
   - Distribution and correlation analysis
   - Loss ratio analysis by geography and demographics
   - Outlier detection

3. **Visual Insights**
   - Key visualizations to highlight findings

## 🔧 Technologies Used

- Python 3.10+
- Jupyter Notebook
- Pandas, NumPy, Matplotlib, Seaborn, Plotly
- GitHub Actions (CI/CD)
- VS Code / JupyterLab

## 📂 Folder Structure

AlphaCare-Insurance-Analytics/
│
├── data/ # Raw and processed datasets
│ └── insurance_data.csv
│
├── notebooks/ # All analysis notebooks
│ ├── data_summary.ipynb
│ ├── eda_univariate.ipynb
│ ├── eda_bivariate.ipynb
│ └── visuals.ipynb
│
├── plots/ # Generated figures and plots
│ └── loss_ratio_by_province.png
│
├── .github/ # CI/CD GitHub Actions workflows
│ └── workflows/
│ └── python-ci.yml
│
├── README.md # Project documentation (you are here)
└── requirements.txt # Python dependencies



## 📦 Setup

```bash
# Clone the repository
git clone https://github.com/your-username/acis-insurance-analysis.git
cd acis-insurance-analysis

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
# 📦 Reproducible Data Pipeline with DVC

This repository sets up a reproducible, version-controlled data pipeline using [Data Version Control (DVC)](https://dvc.org/), ensuring auditability, traceability, and compliance — essential for finance and insurance workflows.

---

## 🧾 Objective

Establish a transparent and auditable pipeline for insurance data analysis by:

- Tracking datasets using DVC.
- Setting up local storage for versioned data.
- Committing metadata to Git for reproducibility.
- Pushing dataset versions to a local remote.

---

## 📁 Folder Structure

.
├── data/ # Folder containing datasets (tracked by DVC)
├── .dvc/ # DVC metadata files
├── .dvcignore # DVC ignore config
├── README.md # This file
├── .gitignore # Git ignore config
├── requirements.txt # Python dependencies
└── ...

---

## ⚙️ Setup Instructions

### 1. Install DVC

pip install dvc
2. Initialize DVC in Your Project

dvc init
git add .dvc .dvcignore .gitignore
git commit -m "Initialize DVC tracking"

3. Add Local Remote Storage

mkdir -p /path/to/local/storage
dvc remote add -d localstorage /path/to/local/storage
git add .dvc/config
git commit -m "Configure DVC local remote storage"

4. Track Dataset with DVC

dvc add data/insurance_data.txt
git add data/insurance_data.txt.dvc
git commit -m "Track insurance dataset with DVC"

5. Push Dataset to Local Remote
dvc push

**📌 Tasks Completed (Task 3) 
## **Tasks Completed (Task 3)**

This phase focused on statistically validating business hypotheses using A/B testing methods:

- **Metrics Defined**:
  - **Claim Frequency**: Proportion of policies with at least one claim.
  - **Claim Severity**: Average cost of a claim.
  - **Margin**: TotalPremium - TotalClaims.

- **Hypotheses Tested**:
  - H₀: No risk differences across provinces ✅
  - H₀: No risk differences between zip codes ✅
  - H₀: No margin difference between zip codes ✅
  - H₀: No significant gender-based risk difference ✅

- **Statistical Tests Used**:
  - One-way ANOVA for provinces
  - t-tests for gender and margin comparisons
  - Visualizations for comparison of distributions

- **Key Findings**:
  - Significant risk variation exists across provinces and gender.
  - Certain zip codes show significantly different margins.
  - These findings support geographic and demographic segmentation for targeted premium adjustments.
