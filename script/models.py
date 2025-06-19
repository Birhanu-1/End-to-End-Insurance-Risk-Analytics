
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import shap
import warnings
from typing import Tuple, Union
import matplotlib.pyplot as plt
# Configure warnings and matplotlib
warnings.filterwarnings("ignore", category=UserWarning)
plt.style.use('ggplot')


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess data from a file.
    
    Args:
        file_path: Path to the data file (tab-separated txt file)
        
    Returns:
        Preprocessed DataFrame
    """
    try:
        data = pd.read_csv(file_path, delimiter='|')
        
        # Handle missing values
        numeric_cols = data.select_dtypes(include=np.number).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
        data[categorical_cols] = data[categorical_cols].fillna("Missing")
        
        # Create target variable
        data['HasClaim'] = (data['TotalClaims'] > 0).astype(int)
        
        return data
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables using one-hot encoding.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with encoded categorical variables
    """
    # Make a copy to avoid modifying original dataframe
    df_encoded = df.copy()
    
    # Identify categorical columns (object and category dtypes)
    cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    
    if len(cat_cols) == 0:
        return df_encoded
    
    # Convert all categorical columns to strings
    df_encoded[cat_cols] = df_encoded[cat_cols].astype(str)
    
    # Initialize and fit the encoder
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_data = encoder.fit_transform(df_encoded[cat_cols])
    
    # Create DataFrame with encoded features
    encoded_df = pd.DataFrame(encoded_data,
                             columns=encoder.get_feature_names_out(cat_cols),
                             index=df_encoded.index)
    
    # Combine with numeric data
    numeric_df = df_encoded.select_dtypes(exclude=['object', 'category'])
    return pd.concat([numeric_df, encoded_df], axis=1)

def prepare_datasets(data: pd.DataFrame) -> Tuple:
    """
    Prepare datasets for both regression and classification tasks.
    
    Args:
        data: Processed DataFrame
        
    Returns:
        Tuple containing train/test splits for both tasks
    """
    data_encoded = encode_categoricals(data)
    
    # Severity prediction (regression) - only for claims > 0
    severity_data = data_encoded[data_encoded['TotalClaims'] > 0].copy()
    X_sev = severity_data.drop(columns=['TotalClaims', 'HasClaim'])
    y_sev = severity_data['TotalClaims']
    X_sev_train, X_sev_test, y_sev_train, y_sev_test = train_test_split(
        X_sev, y_sev, test_size=0.2, random_state=42
    )
    
    # Claim probability prediction (classification)
    X_clf = data_encoded.drop(columns=['TotalClaims', 'HasClaim'])
    y_clf = data_encoded['HasClaim']
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    return (X_sev_train, X_sev_test, y_sev_train, y_sev_test, 
            X_clf_train, X_clf_test, y_clf_train, y_clf_test)


def train_severity_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    """
    Train XGBoost regression model for claim severity prediction.
    """
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_classification_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier for claim probability prediction.
    """
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc'
    )
    model.fit(X_train, y_train)
    return model


def evaluate_regression(model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
    """
    Evaluate regression model and print metrics.
    """
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print("\n--- Claim Severity Evaluation ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    return preds


def evaluate_classification(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
    """
    Evaluate classification model and print metrics.
    """
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Claim Probability Evaluation ---")
    print(classification_report(y_test, preds))
    print(f"ROC AUC: {roc_auc_score(y_test, proba):.4f}")
    
    return proba


def calculate_premium(prob_claim: Union[float, np.ndarray], 
                     avg_severity: float, 
                     loading: float = 200) -> Union[float, np.ndarray]:
    """
    Calculate insurance premium based on predicted probability and average severity.
    
    Premium = (Probability of Claim × Average Severity) + Loading Factor
    """
    return prob_claim * avg_severity + loading


def explain_model(model: Union[xgb.XGBRegressor, xgb.XGBClassifier], 
                 X_test: pd.DataFrame, 
                 title: str) -> None:
    """
    Generate SHAP explanations for model predictions.
    """
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)
        
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title(f"SHAP Summary - {title}")
        plt.tight_layout()
        plt.savefig(f"shap_summary_{title.lower().replace(' ', '_')}.png")
        plt.close()
        
        print(f"\nSHAP summary plot saved for {title} model")
    except Exception as e:
        print(f"Error generating SHAP explanation: {str(e)}")


def main(file_path: str) -> None:
    """
    Main execution function for the insurance pricing model.
    """
    print("Starting insurance pricing model...")
    
    # Load and prepare data
    print("\nLoading and preprocessing data...")
    data = load_data(file_path)
    
    print("\nPreparing datasets...")
    (X_sev_train, X_sev_test, y_sev_train, y_sev_test,
     X_clf_train, X_clf_test, y_clf_train, y_clf_test) = prepare_datasets(data)
    
    # Train models
    print("\nTraining severity model...")
    sev_model = train_severity_model(X_sev_train, y_sev_train)
    
    print("\nTraining classification model...")
    clf_model = train_classification_model(X_clf_train, y_clf_train)
    
    # Evaluate models
    print("\nEvaluating models...")
    sev_preds = evaluate_regression(sev_model, X_sev_test, y_sev_test)
    clf_probs = evaluate_classification(clf_model, X_clf_test, y_clf_test)
    
    # Calculate premiums
    avg_severity = y_sev_train.mean()
    premiums = calculate_premium(clf_probs, avg_severity)
    
    print("\n--- Premium Calculation ---")
    print(f"Average claim severity: ${avg_severity:,.2f}")
    print("\nSample predicted premiums:")
    for i, premium in enumerate(premiums[:5]):
        print(f"Policy {i+1}: ${premium:,.2f}")
    
    # Explain models
    print("\nGenerating model explanations...")
    explain_model(sev_model, X_sev_test, "Severity Model")
    explain_model(clf_model, X_clf_test, "Claim Probability Model")
    
    print("\nModel pipeline completed successfully!")


if __name__ == "__main__":
    # For Jupyter notebook usage
    data_file = "../data/MachineLearningRating_v3.txt"  # Set your file path here
    main(data_file)