import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load preprocessed dataset."""
    return pd.read_csv(file_path)

def train_model(X_train, y_train):
    """Train linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def save_model(model, file_path):
    """Save trained model."""
    joblib.dump(model, file_path)

if __name__ == "__main__":
    # Load preprocessed data
    data = load_data("data/processed/cleaned_housing_data.csv")
    
    # Define features and target
    X = data.drop("price", axis=1)  # All columns except 'price'
    y = data["price"]
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
    
    # Train model on training data
    model = train_model(X_train, y_train)
    
    # Evaluate model on test data
    mse, r2 = evaluate_model(model, X_test, y_test)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Save model
    save_model(model, "models/linear_regression_model.pkl")