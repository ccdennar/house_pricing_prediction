## House Price Prediction Project
# Overview
This project builds a machine learning pipeline to predict house prices using the Ames Housing dataset. It includes data preprocessing, exploratory data analysis (EDA), model training, and evaluation. A linear regression model is used, with preprocessing steps to handle missing values, encode categorical features, and scale numerical features. The project showcases skills in data cleaning, feature engineering, and model evaluation, ideal for a data science portfolio.

# Features
i. Data Preprocessing: Imputes missing values, one-hot encodes categorical features, and scales numerical features.

ii. Exploratory Data Analysis: Visualizes feature distributions and relationships via a Jupyter notebook.

iii. Model Training: Trains a linear regression model using scikit-learn.

iv. Evaluation: Computes Mean Squared Error (MSE) and R² score on a test set.

v. Modular Design: Separates preprocessing (preprocess.py) and modeling (model.py) scripts.

# Dataset
The Ames Housing dataset contains 80 features (numerical and categorical) and the target variable SalePrice. It is stored in data/raw/housing_data.csv.

## Project Structure

house-price-prediction/
├── data/
│   ├── raw/
│   │   └── housing_data.csv    # Raw dataset
│   ├── processed/
│   │   └── cleaned_housing_data.csv    # Preprocessed dataset (generated)
├── models/
│   └── linear_regression_model.pkl    # Trained model (generated)
├── notebooks/
│   └── eda.ipynb                    # EDA notebook
├── src/
│   ├── preprocess.py                # Preprocessing script
│   ├── model.py                     # Modeling script
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation


## 