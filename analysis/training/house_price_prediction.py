import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn
import mlflow.lightgbm

from sklearn.datasets import fetch_california_housing # ONLY TESTING

# Set MLflow experiment
mlflow.set_experiment("house_price_prediction")

def load_data(data_path):
    """Load and preprocess the dataset"""
    # df = pd.read_csv(data_path)

    df =  fetch_california_housing(as_frame=True).frame # ONLY TESTING
    # Assuming your target variable is named 'price'
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    # Handle categorical variables (if any)
    X = pd.get_dummies(X, drop_first=True)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluation metrics function
def evaluate_model(model, X_test, y_test):
    """Calculate regression metrics"""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return {
        "rmse": rmse,
        "mse": mse,
        "mae": mae,
        "r2": r2
    }

# Training function for sklearn Random Forest
def train_sklearn_rf(X_train, X_test, y_train, y_test, params):
    with mlflow.start_run(run_name="sklearn_rf"):
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        
        # Evaluate and log metrics
        metrics = evaluate_model(rf, X_test, y_test)
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(rf, "model")
        
        return rf, metrics

# Training function for LightGBM
def train_lightgbm(X_train, X_test, y_train, y_test, params):
    with mlflow.start_run(run_name="lightgbm"):
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        lgb = LGBMRegressor(**params)
        lgb.fit(X_train, y_train)
        
        # Evaluate and log metrics
        metrics = evaluate_model(lgb, X_test, y_test)
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.lightgbm.log_model(lgb, "model")
        
        return lgb, metrics

# Example usage in notebook cells:
# Cell 1: Load data
X_train, X_test, y_train, y_test = load_data("your_data.csv")

# Cell 2: Define parameters for each model
sklearn_params = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42
}

lightgbm_params = {
    "n_estimators": 100,
    "max_depth": 10,
    "num_leaves": 31,
    "random_state": 42
}

# Cell 3: Train sklearn Random Forest
sklearn_model, sklearn_metrics = train_sklearn_rf(
    X_train, X_test, y_train, y_test, sklearn_params
)
print("Sklearn RF Metrics:", sklearn_metrics)

# Cell 4: Train LightGBM
lightgbm_model, lightgbm_metrics = train_lightgbm(
    X_train, X_test, y_train, y_test, lightgbm_params
)
print("LightGBM Metrics:", lightgbm_metrics)