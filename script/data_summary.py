

"""1.2 Project Planning - EDA & Stats
Tasks: 
- Data Understanding
- Exploratory Data Analysis (EDA)

This notebook focuses on:
- Data Summarization:
  * Descriptive Statistics: Variability of numerical features such as TotalPremium, TotalClaim
  * Data Structure: dtypes and format check
- Data Quality Assessment:
  * Missing values overview
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Load data (update path as needed)
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, delimiter='\t')
    return df

# Summarize data
def summarize_data(df: pd.DataFrame):
    print("\n--- Descriptive Statistics ---")
    display(df.describe(include='all'))
    
    print("\n--- Data Types ---")
    print(df.dtypes)

# Assess missing values
def assess_data_quality(df: pd.DataFrame):
    print("\n--- Missing Value Summary ---")
    missing = df.isnull().sum()
    percent = (df.isnull().mean() * 100)
    result = pd.DataFrame({'MissingCount': missing, 'MissingPercent': percent})
    display(result[result.MissingCount > 0].sort_values(by='MissingCount', ascending=False))

# Run functions
if __name__ == "__main__":
    filepath = "../data/MachineLearningRating_v3.txt"
    df = load_data(filepath)
    summarize_data(df)
    assess_data_quality(df)