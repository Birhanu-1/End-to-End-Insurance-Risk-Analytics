# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime

def load_data(file_path):
    """Load and preprocess the raw insurance data"""
    # Load data with proper parsing
    data = pd.read_csv(file_path, sep='|', parse_dates=['TransactionMonth'])
    
    # Clean column names (remove leading/trailing spaces)
    data.columns = data.columns.str.strip()
    
    # Convert TotalClaims to numeric, handling special cases
    data['TotalClaims'] = pd.to_numeric(data['TotalClaims'], errors='coerce').fillna(0)
    
    # Convert CalculatedPremiumPerTerm to numeric
    data['CalculatedPremiumPerTerm'] = pd.to_numeric(data['CalculatedPremiumPerTerm'], errors='coerce')
    
    # Calculate vehicle age
    current_year = datetime.now().year
    data['VehicleAge'] = current_year - data['RegistrationYear']
    
    # Create claim flag
    data['HasClaim'] = (data['TotalClaims'] > 0).astype(int)
    
    return data

def prepare_severity_data(data):
    """Prepare data for claim severity modeling (only policies with claims)"""
    severity_data = data[data['TotalClaims'] > 0].copy()
    
    # Define features and target
    features = severity_data.drop(columns=['TotalClaims', 'HasClaim', 'UnderwrittenCoverID', 'PolicyID'])
    target = severity_data['TotalClaims']
    
    return features, target

def prepare_probability_data(data):
    """Prepare data for claim probability modeling"""
    features = data.drop(columns=['TotalClaims', 'HasClaim', 'UnderwrittenCoverID', 'PolicyID'])
    target = data['HasClaim']
    
    return features, target

def get_preprocessor(X):
    """Create preprocessing pipeline based on data types"""
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(exclude=['number']).columns
    
    # Numeric preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor

def main():
    # Load and preprocess data
    data = load_data('../data/MachineLearningRating_v3.txt')
    
    # Prepare severity data
    X_sev, y_sev = prepare_severity_data(data)
    preprocessor_sev = get_preprocessor(X_sev)
    
    # Prepare probability data
    X_prob, y_prob = prepare_probability_data(data)
    preprocessor_prob = get_preprocessor(X_prob)
    
    # Split data (80% train, 20% test)
    X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
        X_sev, y_sev, test_size=0.2, random_state=42
    )
    
    X_train_prob, X_test_prob, y_train_prob, y_test_prob = train_test_split(
        X_prob, y_prob, test_size=0.2, random_state=42
    )
    
    # Save preprocessed data
    np.savez('../data/preprocessed_data.npz',
         X_train_sev=X_train_sev.values,
         X_test_sev=X_test_sev.values,
         y_train_sev=y_train_sev.values,
         y_test_sev=y_test_sev.values,
         X_train_prob=X_train_prob.values,
         X_test_prob=X_test_prob.values,
         y_train_prob=y_train_prob.values,
         y_test_prob=y_test_prob.values,
         columns_sev=X_train_sev.columns.values,  # Save column names
         columns_prob=X_train_prob.columns.values,
         preprocessor_sev=preprocessor_sev,
         preprocessor_prob=preprocessor_prob)
    
    print("Data preprocessing completed successfully.")

if __name__ == "__main__":
    main()