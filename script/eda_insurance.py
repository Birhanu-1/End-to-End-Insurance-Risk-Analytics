import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Data loaded. Shape: {df.shape}")
    return df

def summarize_data(df):
    print("Summary Statistics:\n", df.describe())
    print("\nData Types:\n", df.dtypes)

def check_missing_values(df):
    missing = df.isnull().sum()
    print("\nMissing Values:\n", missing[missing > 0])

def univariate_analysis(df):
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in num_cols[:5]:  # Limit to 5 for quick viewing
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()

    for col in cat_cols[:3]:  # Limit to 3
        plt.figure(figsize=(6, 4))
        df[col].value_counts().plot(kind='bar')
        plt.title(f"Frequency of {col}")
        plt.xticks(rotation=45)
        plt.show()

def bivariate_analysis(df):
    if 'TotalClaims' in df.columns and 'TotalPremium' in df.columns:
        df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x='TotalPremium', y='TotalClaims', data=df)
        plt.title("Total Premium vs Total Claims")
        plt.show()

        corr_matrix = df[['TotalPremium', 'TotalClaims', 'LossRatio']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.show()

def geography_comparison(df):
    if 'Province' in df.columns and 'TotalClaims' in df.columns:
        province_stats = df.groupby('Province')[['TotalClaims', 'TotalPremium']].mean().reset_index()
        plt.figure(figsize=(8, 5))
        sns.barplot(x='TotalClaims', y='Province', data=province_stats, color='skyblue')
        plt.title("Avg Total Claims by Province")
        plt.show()

def detect_outliers(df):
    if 'TotalClaims' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df['TotalClaims'])
        plt.title("Boxplot of Total Claims")
        plt.show()

    if 'CustomValueEstimate' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df['CustomValueEstimate'])
        plt.title("Boxplot of Custom Value Estimate")
        plt.show()

def creative_visuals(df):
    # Plot 1: Claims by VehicleType
    if 'VehicleType' in df.columns:
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=df, x='VehicleType', y='TotalClaims')
        plt.title("Claim Distribution by Vehicle Type")
        plt.xticks(rotation=45)
        plt.show()

    # Plot 2: Monthly trend
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
        trend = df.groupby(df['TransactionMonth'].dt.to_period('M')).agg({'TotalClaims': 'sum', 'TotalPremium': 'sum'}).reset_index()
        trend['TransactionMonth'] = trend['TransactionMonth'].astype(str)

        plt.figure(figsize=(10, 5))
        sns.lineplot(data=trend, x='TransactionMonth', y='TotalClaims', label='TotalClaims')
        sns.lineplot(data=trend, x='TransactionMonth', y='TotalPremium', label='TotalPremium')
        plt.xticks(rotation=45)
        plt.title("Monthly Total Claims & Premium")
        plt.legend()
        plt.show()

    # Plot 3: Loss Ratio Distribution
    if 'LossRatio' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df['LossRatio'], bins=30, kde=True)
        plt.title("Loss Ratio Distribution")
        plt.show()

def main():
    file_path = "insurance_data.csv"  # Replace with your CSV file path
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = load_data(file_path)
    summarize_data(df)
    check_missing_values(df)
    univariate_analysis(df)
    bivariate_analysis(df)
    geography_comparison(df)
    detect_outliers(df)
    creative_visuals(df)

if __name__ == "__main__":
    main()
