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


