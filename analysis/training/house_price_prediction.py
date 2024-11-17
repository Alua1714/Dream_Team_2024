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
from joblib import dump, load
import os
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional, Any
import matplotlib as plt
import time


class BaseModel(ABC):
    """Abstract base class for all models in the framework"""
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name for logging"""
        pass

class LightGBMModel(BaseModel):
    """LightGBM model implementation"""
    
    def __init__(self, params: Dict[str, Any]):
        self.model = LGBMRegressor(**params)
        self.params = params
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(X_train, y_train)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def get_model_name(self) -> str:
        return "LightGBM"
    
    def get_feature_importance(self) -> Dict[str, float]:
        return dict(zip(self.model.feature_name_, self.model.feature_importances_))

class TimeSeriesCV:
    """Time series cross-validation with expanding window"""
    
    def __init__(self, min_training_months: int = 1, forecast_months: int = 2):
        self.min_training_months = min_training_months
        self.forecast_months = forecast_months
    
    def split(self, X: pd.DataFrame, date_column: str = 'Listing.Dates.CloseDate') -> List[Tuple[np.ndarray, np.ndarray]]:
        # Convert dates if they're strings
        if isinstance(X[date_column].iloc[0], str):
            X[date_column] = pd.to_datetime(X[date_column])
            
        # Create month_year column
        X['month_year'] = X[date_column].dt.to_period('M')
        
        # Get sorted unique months
        unique_months = sorted(X['month_year'].unique())
        n_months = len(unique_months)
        
        print(f"Total number of months: {n_months}")
        print(f"Min training months: {self.min_training_months}")
        print(f"Forecast months: {self.forecast_months}")
        
        if n_months < self.min_training_months + self.forecast_months:
            raise ValueError("Not enough months in the dataset")
        
        splits = []
        for i in range(self.min_training_months, n_months - self.forecast_months + 1):
            train_months = unique_months[:i]
            val_months = unique_months[i:i + self.forecast_months]
            
            # Get boolean masks for train and validation sets
            train_mask = X['month_year'].isin(train_months)
            val_mask = X['month_year'].isin(val_months)
            
            # Convert boolean masks to integer indices
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            
            print(f"\nFold {len(splits)}:")
            print(f"Train months: {train_months[0]} to {train_months[-1]}")
            print(f"Val months: {val_months[0]} to {val_months[-1]}")
            print(f"Train indices range: {train_idx.min()} to {train_idx.max()}")
            print(f"Val indices range: {val_idx.min()} to {val_idx.max()}")
            print(f"X shape: {X.shape}")
            
            # Verify indices are within bounds
            if train_idx.max() >= len(X) or val_idx.max() >= len(X):
                print("Warning: Indices out of bounds detected!")
                continue
            
            splits.append((train_idx, val_idx))
        
        if not splits:
            raise ValueError("No valid splits were generated!")
            
        return splits

class ModelPersistence:
    """Handle model saving and loading"""
    
    def __init__(self, base_dir: str = 'models'):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def save_model(self, model: BaseModel, scaler: Optional[StandardScaler] = None, 
                  metrics: Optional[Dict] = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.base_dir, model.get_model_name(), timestamp)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        dump(model, os.path.join(model_dir, 'model.joblib'))
        
        if scaler is not None:
            dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
        
        if metrics is not None:
            dump(metrics, os.path.join(model_dir, 'metrics.joblib'))
        
        return timestamp
    
    def load_model(self, model_name: str, timestamp: str) -> Tuple[BaseModel, Optional[StandardScaler], Optional[Dict]]:
        model_dir = os.path.join(self.base_dir, model_name, timestamp)
        
        model = load(os.path.join(model_dir, 'model.joblib'))
        
        scaler = None
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        if os.path.exists(scaler_path):
            scaler = load(scaler_path)
        
        metrics = None
        metrics_path = os.path.join(model_dir, 'metrics.joblib')
        if os.path.exists(metrics_path):
            metrics = load(metrics_path)
        
        return model, scaler, metrics

class ModelEvaluator:
    """Handle model evaluation and metrics calculation"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    @staticmethod
    def evaluate_single_prediction(X: pd.DataFrame, y: pd.Series, model: BaseModel, 
                                 sample_index: int = 0) -> Dict:
        
        sample = X.iloc[sample_index:sample_index+1]
        actual_value = y.iloc[sample_index]
        
        prediction = model.predict(sample)[0]
        error = prediction - actual_value
        percentage_error = (error / actual_value) * 100
        
        return {
            'prediction': prediction,
            'actual': actual_value,
            'error': error,
            'percentage_error': percentage_error
        }

class HousePricePredictor:
    """Main class for house price prediction workflow"""
    
    def __init__(self, model: BaseModel, experiment_name: str = "house_price_prediction"):
        self.model = model
        self.persistence = ModelPersistence()
        self.evaluator = ModelEvaluator()
        self.cv = TimeSeriesCV()
        
        # Set up MLflow tracking
        mlflow_dir = os.path.abspath("mlruns")
        if not os.path.exists(mlflow_dir):
            os.makedirs(mlflow_dir)
            
        # Set the tracking URI to the absolute path
        mlflow.set_tracking_uri(f"file:{mlflow_dir}")
        
        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=os.path.join(mlflow_dir, experiment_name)
            )
        else:
            experiment_id = experiment.experiment_id
            
        mlflow.set_experiment(experiment_name)
        
        print(f"\nMLflow Configuration:")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Experiment Name: {experiment_name}")
        print(f"Experiment ID: {experiment_id}")

    def prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load and preprocess data"""
        df = pd.read_csv(data_path)
        
        # Convert date column to datetime
        df['Listing.Dates.CloseDate'] = pd.to_datetime(df['Listing.Dates.CloseDate'])

        X = df.drop('Listing.Price.ClosePrice', axis=1)
        y = df['Listing.Price.ClosePrice']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train model with time series CV and log results with console output"""
        results = {'fold_metrics': [], 'predictions': []}
        
        print("\n" + "="*50)
        print("Starting Training and Evaluation")
        print("="*50)
        
        with mlflow.start_run(run_name=f"training_{self.model.get_model_name()}"):
            splits = self.cv.split(X)
            print(f"\nNumber of splits generated: {len(splits)}")
            
            # Initialize aggregate metrics
            total_metrics = {
                'rmse': [],
                'mae': [],
                'r2': []
            }
            
            for fold, (train_idx, val_idx) in enumerate(splits):
                print(f"\nFold {fold + 1}/{len(splits)}")
                print("-"*30)
                
                # Prepare fold data
                X_train_fold = X.iloc[train_idx].copy()
                X_val_fold = X.iloc[val_idx].copy()
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]
                
                # Drop date columns
                date_cols = ['Listing.Dates.CloseDate', 'month_year']
                X_train_fold = X_train_fold.drop(columns=[col for col in date_cols if col in X_train_fold.columns])
                X_val_fold = X_val_fold.drop(columns=[col for col in date_cols if col in X_val_fold.columns])
                
                print(f"Training samples: {len(X_train_fold)}")
                print(f"Validation samples: {len(X_val_fold)}")
                
                # Train and predict
                self.model.train(X_train_fold, y_train_fold)
                val_pred = self.model.predict(X_val_fold)
                
                # Calculate metrics
                metrics = self.evaluator.calculate_metrics(y_val_fold, val_pred)
                metrics['fold'] = fold
                
                # Print fold metrics
                print("\nFold Metrics:")
                print(f"RMSE: ${metrics['rmse']:,.2f}")
                print(f"MAE:  ${metrics['mae']:,.2f}")
                print(f"R2:   {metrics['r2']:.4f}")
                
                # Accumulate metrics for averaging
                for metric_name in ['rmse', 'mae', 'r2']:
                    total_metrics[metric_name].append(metrics[metric_name])
                
                # Log fold metrics to MLflow
                try:
                    mlflow.log_metrics({
                        f"fold_{fold}_rmse": metrics['rmse'],
                        f"fold_{fold}_mae": metrics['mae'],
                        f"fold_{fold}_r2": metrics['r2']
                    })
                except Exception as e:
                    print(f"Warning: Failed to log metrics to MLflow: {str(e)}")
                
                results['fold_metrics'].append(metrics)
                results['predictions'].append({
                    'fold': fold,
                    'true_values': y_val_fold.tolist(),
                    'predictions': val_pred.tolist()
                })

                if fold == 8:
                    print('Getting rellevant features...')
                    self.get_rellevant_features(X_val_fold)

            
            # Calculate and print average metrics
            avg_metrics = {
                metric: np.mean(values) for metric, values in total_metrics.items()
            }
            
            print("\n" + "="*50)
            print("Average Metrics Across All Folds:")
            print("="*50)
            print(f"Average RMSE: ${avg_metrics['rmse']:,.2f}")
            print(f"Average MAE:  ${avg_metrics['mae']:,.2f}")
            print(f"Average R2:   {avg_metrics['r2']:.4f}")
            
            # Print metric ranges
            print("\nMetric Ranges:")
            for metric in ['rmse', 'mae', 'r2']:
                min_val = min(total_metrics[metric])
                max_val = max(total_metrics[metric])
                if metric in ['rmse', 'mae']:
                    print(f"{metric.upper()}: ${min_val:,.2f} to ${max_val:,.2f}")
                else:
                    print(f"{metric.upper()}: {min_val:.4f} to {max_val:.4f}")
            
            # Try to log average metrics to MLflow
            try:
                mlflow.log_metrics({
                    f"avg_{metric}": value for metric, value in avg_metrics.items()
                })
            except Exception as e:
                print(f"Warning: Failed to log average metrics to MLflow: {str(e)}")
        
        return results
    
    def get_rellevant_features(self, X_test, random_state = 4):
        # Extracting feature importances from the model (if available)
        model = self.model
        feature_names = list(X_test.columns)

        # Checking if the model has feature_importances_ or coefficients
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            importances = None
            print("The model does not have 'feature_importances_' or 'coef_' attributes")

        if importances is not None:
            # Creating a DataFrame for feature importances
            feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
            
            # Saving and logging feature importances as CSV
            feat_imp_csv = 'feature_importance.csv'
            feat_imp_df.to_csv(feat_imp_csv, index=False)
            mlflow.log_artifact(feat_imp_csv)

            # Plotting and saving feature importances
            plt.figure(figsize=(10, 6))
            plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'])
            plt.gca().invert_yaxis()
            plt.xlabel('Importance')
            plt.title('Feature Importances')
            plt.tight_layout()
            
            # Saving and log the plot
            feat_imp_png = 'feature_importance.png'
            plt.savefig(feat_imp_png)
            plt.close()
            mlflow.log_artifact(feat_imp_png)

def main():
    # Clear existing mlruns directory if it exists (optional, remove if you want to keep history)

    mlruns_dir = "mlruns"
    if not os.path.exists(mlruns_dir):
        os.makedirs(mlruns_dir)

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), 'dataset'))
    train_file = os.path.abspath(os.path.join(parent_dir, 'df_train.csv'))
    
    # Initialize model with parameters
    lgb_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "min_child_samples": 20,
        "min_split_gain": 0.1,
        "random_state": 42,
        "verbose": -1
    }
    model = LightGBMModel(lgb_params)
    
    # Initialize predictor with experiment name
    predictor = HousePricePredictor(model, "house_price_experiment")
    predictor.cv = TimeSeriesCV(min_training_months=3, forecast_months=1)
    
    # Print starting message
    print("\nStarting House Price Prediction")
    print("="*50)
    print("Model Parameters:")
    for param, value in lgb_params.items():
        print(f"{param}: {value}")
    
    # Prepare data
    X_train, X_test, y_train, y_test = predictor.prepare_data(train_file)
    
    print("\nData Shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Train and evaluate
    results = predictor.train_and_evaluate(X_train, y_train)

    predictor.persistence.save_model(model)

    print("\nMLflow UI can be started with:")
    print("mlflow ui --backend-store-uri file:./mlruns")

if __name__ == "__main__":
    main()