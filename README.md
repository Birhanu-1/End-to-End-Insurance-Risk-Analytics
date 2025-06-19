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
# Clone the repository
git clone https://github.com/Birhanu-1/End_to_End_Insurance_Risk_Analytics.git
cd End_to_End_Insurance_Risk_Analytics

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt




## **📌 𝗧𝗔𝗦𝗞𝗦 𝗖𝗢𝗠𝗣𝗟𝗘𝗧𝗘𝗗 (𝗧𝗔𝗦𝗞 𝟮)**
** 🧾 Objective

** Establish a transparent and auditable pipeline for insurance data analysis by:

**- Tracking datasets using DVC.
**- Setting up local storage for versioned data.
**- Committing metadata to Git for reproducibility.
**- Pushing dataset versions to a local remote. 

** 📁 Folder Structure
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

mkdir -p D:\dvc-remote-storage
dvc remote add -d localstorage D:\dvc-remote-storage
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
=======
# Task-4: Risk-Based Insurance Pricing System

## 📌 Objective
This task focuses on building and evaluating machine learning models for dynamic, risk-based insurance pricing. The goal is to predict:
1. **Claim Severity** for policies that have a claim (`TotalClaims > 0`)
2. **Claim Probability** for all policies
3. **Premium Optimization** using a risk-based formula:
Premium = (Probability of Claim × Predicted Claim Severity) + Expense Loading + Profit Margin



---

## 🧪 Project Structure

### `data_preprocessing.ipynb`
- Loads and cleans the raw dataset
- Handles missing values (median for numeric, 'Missing' for categorical)
- Performs one-hot encoding on categorical variables
- Splits data for:
- Regression: Claim Severity
- Classification: Claim Probability
- Saves train-test splits using `joblib`

### `model_training_and_evaluation.ipynb`
- Trains multiple models:
- **Regression**: `LinearRegression`, `RandomForest`, `XGBoost`
- **Classification**: `RandomForest`, `XGBoost`
- Evaluates models using:
- **Regression**: RMSE, R²
- **Classification**: Accuracy, Precision, Recall, F1-score, AUC
- Calculates premiums using predicted probabilities and severities
- Interprets best-performing models using **SHAP** values

---

## 🧠 Modeling Techniques

### Claim Severity (Regression)
- **Target**: `TotalClaims` (for policies with `TotalClaims > 0`)
- **Metrics**: RMSE, R²
- **Models**:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

### Claim Probability (Classification)
- **Target**: `HasClaim` (binary: 0/1)
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Models**:
- Random Forest Classifier
- XGBoost Classifier

---

## 📊 Model Interpretability

### SHAP Analysis
- Top 5–10 most influential features visualized via SHAP summary plots
- Business insights extracted from SHAP values:
> For example: "For every year older a vehicle is, the predicted claim amount increases by X Rand, holding other factors constant."

---

## 🛠️ Setup Instructions

```bash
# Clone the repo and navigate to task-4
git checkout -b task-4

# Install dependencies
pip install -r requirements.txt