import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """Load raw dataset."""
    return pd.read_csv(file_path)

def handle_missing_values(df):
    """Impute missing values: median for numerical, 'None' for categorical."""
    # Numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_imputer = SimpleImputer(strategy='constant', fill_value='None')
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
    
    return df

def scale_features(df, target_col='price'):
    """Scale numerical features using StandardScaler, excluding target."""
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # Exclude target column from scaling
    numerical_cols = [col for col in numerical_cols if col != target_col]
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def preprocess_categorical(df):
    """One-hot encode categorical features."""
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

def save_data(df, file_path):
    """Save preprocessed data."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    raw_data_path = "data/raw/housing_data.csv"
    processed_data_path = "data/processed/cleaned_housing_data.csv"

    # Load data
    df = load_data(raw_data_path)
    
    # Rename target column
    df = df.rename(columns={'SalePrice': 'price'})
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # One-hot encode categorical features
    df = preprocess_categorical(df)
    
    # Scale numerical features
    df = scale_features(df, target_col='price')
    
    # Save preprocessed data
    save_data(df, processed_data_path)
    print(f"Preprocessed data saved to {processed_data_path}")