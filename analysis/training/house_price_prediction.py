import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from joblib import dump, load
import os
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional, Any
import matplotlib.pylab as plt
import time
from sklearn.model_selection import learning_curve


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

class EnsembleLightGBMModel(BaseModel):
    """Ensemble of LightGBM models with different parameters and weighted averaging"""
    
    def __init__(self, model_configs: List[Dict[str, Any]], weights: List[float] = None):
        """
        Initialize ensemble with multiple model configurations
        
        Args:
            model_configs: List of dictionaries containing parameters for each model
            weights: List of weights for ensemble averaging. Must sum to 1.
        """
        if weights is None:
            self.weights = [1/len(model_configs)] * len(model_configs)
        else:
            if len(weights) != len(model_configs):
                raise ValueError("Number of weights must match number of models")
            if abs(sum(weights) - 1.0) > 1e-5:
                raise ValueError("Weights must sum to 1")
            self.weights = weights
            
        self.models = [LGBMRegressor(**config) for config in model_configs]
        self.feature_importances_ = None
        self.feature_name_ = None
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train all models in the ensemble"""
        self.feature_name_ = X_train.columns.tolist()
        
        print("\nTraining Ensemble Models:")
        for i, model in enumerate(self.models, 1):
            print(f"\nTraining Model {i} (Weight: {self.weights[i-1]:.2f})")
            model.fit(X_train, y_train, feature_name=self.feature_name_)
            
            # Calculate training metrics for each model
            y_pred = model.predict(X_train)
            mae = mean_absolute_error(y_train, y_pred)
            print(f'Model {i} Training MAE: ${mae:,.2f}')
        
        # Combine feature importances with weights
        self.feature_importances_ = np.zeros(len(self.feature_name_))
        for model, weight in zip(self.models, self.weights):
            self.feature_importances_ += weight * model.feature_importances_
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make weighted predictions using all models"""
        predictions = np.zeros(len(X))
        for model, weight in zip(self.models, self.weights):
            predictions += weight * model.predict(X)
        return predictions
    
    def get_model_name(self) -> str:
        return "EnsembleLightGBM"
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get weighted feature importance across all models"""
        if self.feature_importances_ is None or self.feature_name_ is None:
            raise ValueError("Model must be trained before getting feature importance")
        return dict(zip(self.feature_name_, self.feature_importances_))

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
        dump(model, os.path.join(model_dir, 'model.joblib'), protocol=4)
        
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

    def create_engineered_features(self, df):
        df['Polar2'] = pow(df['Polar.Theta'], 2)
        df['Polar3'] = pow(df['Polar.R'], 4)

        # Area ratios
        df['living_to_lot_ratio'] = np.log( df['Structure.LivingArea'] / df['Characteristics.LotSizeSquareFeet'])
        df['below_grade_ratio'] = np.log(df['Structure.BelowGradeFinishedArea'] / df['Structure.LivingArea'])
        
        # Bathroom features
        df['total_bathrooms'] = df['Structure.BathroomsFull'] + 0.5 * df['Structure.BathroomsHalf']
        df['bathroom_density'] = df['total_bathrooms'] / df['Structure.LivingArea']
        
        # Age features
        current_year = 2024
        df['property_age'] = current_year - df['Structure.YearBuilt']
        df['age_bracket'] = pd.cut(df['property_age'], 
                                bins=[0, 10, 20, 30, 40, 50, float('inf')],
                                labels=['0-10', '11-20', '21-30', '31-40', '41-50', '50+'])
        
        # Room metrics
        df['room_density'] = df['Structure.Rooms.RoomsTotal'] / df['Structure.LivingArea']
        df['bedroom_ratio'] = df['Structure.BedroomsTotal'] / df['Structure.Rooms.RoomsTotal']
        
        # Quality scores
        df['interior_quality'] = df[[col for col in df.columns if 'interior' in col.lower()]].mean(axis=1)
    
        return df

    def prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load and preprocess data"""
        df = pd.read_csv(data_path)
        df = self.create_engineered_features(df)

        # top_feat = ['one_hot_Appliances.WineCooler', 'one_hot_ParkingFeatures.ParkingPad', 'one_hot_ParkingFeatures.ElectricVehicleChargingStations', 'one_hot_Appliances.WaterSoftener', 'one_hot_FireplaceFeatures.Bedroom', 'one_hot_View.CreekStream', 'one_hot_OtherStructures.Storage', 'one_hot_Heating.Oil', 'one_hot_LaundryFeatures.CommonArea', 'one_hot_Office', 'one_hot_Levels.OneAndOneHalf', 'one_hot_AssociationAmenities.Barbecue', 'one_hot_OtherStructures.Barns', 'one_hot_golf_course_lot', 'one_hot_infill_lot', 'one_hot_other', 'one_hot_forest_preserve_adjacent', 'one_hot_View.Mountains', 'one_hot_partial_fencing', 'one_hot_creek', 'one_hot_OtherStructures.PoolHouse', 'one_hot_Heating.Propane', 'one_hot_AssociationAmenities.BasketballCourt', 'one_hot_PoolFeatures.AboveGround', 'one_hot_Heating.Ductless', 'one_hot_Appliances.GasWaterHeater', 'one_hot_InteriorOrRoomFeatures.Chandelier', 'one_hot_dual', 'one_hot_total', 'one_hot_window/wall_units_-_2', 'one_hot_rtail', 'one_hot_indus', 'one_hot_chillers', 'one_hot_reverse_cycle', 'one_hot_power_roof_vents', 'one_hot_office_only', 'one_hot_pud', 'one_hot_ceiling_fan(s)', 'one_hot_exhaust_fan', 'one_hot_electric', 'one_hot_ParkingFeatures.Basement', 'one_hot_window/wall_unit_-_1', 'one_hot_wall_sleeve', 'one_hot_window_unit(s)', 'one_hot_high_efficiency_(seer_14+)', 'one_hot_Storage', 'one_hot_ExteriorFeatures.Balcony', 'one_hot_views', 'one_hot_business_opportunity', 'one_hot_wooded', 'one_hot_heat_pump', 'one_hot_partial', 'one_hot_air_curtain', 'one_hot_roof_turbine(s)', 'one_hot_dimensions_to_center_of_road', 'one_hot_window/wall_units_-_3+', 'one_hot_gas', 'one_hot_manuf', 'one_hot_lnc', 'one_hot_offic', 'one_hot_commr', 'one_hot_pmd', 'one_hot_wrhse', 'one_hot_ConstructionMaterials.BoardAndBattenSiding', 'one_hot_pasture', 'one_hot_AssociationAmenities.Storage', 'one_hot_CommunityFeatures.Curbs', 'one_hot_Flooring.Stone', 'one_hot_pie_shaped_lot', 'one_hot_AssociationAmenities.Playground', 'one_hot_backs_to_public_grnd', 'one_hot_level', 'one_hot_Basement.Concrete', 'one_hot_Flooring.Tile', 'one_hot_AssociationAmenities.Laundry', 'one_hot_backs_to_open_grnd', 'one_hot_woven_wire_fence', 'one_hot_nature_preserve_adjacent', 'one_hot_Flooring.Marble', 'one_hot_PatioAndPorchFeatures.Porch', 'one_hot_channel_front', 'one_hot_CommunityFeatures.Golf', 'one_hot_Flooring.Concrete', 'one_hot_InteriorOrRoomFeatures.DryBar', 'one_hot_river_front', 'one_hot_Heating.SpaceHeater', 'one_hot_View.Neighborhood', 'one_hot_Fencing.BackYard', 'one_hot_fenced_yard', 'one_hot_wood_fence', 'one_hot_AssociationAmenities.GameRoom', 'one_hot_LaundryFeatures.GasDryerHookup', 'one_hot_FireplaceFeatures.Kitchen', 'one_hot_Roof.Metal', 'one_hot_OtherStructures.Workshop', 'one_hot_Roof.Tile', 'one_hot_ConstructionMaterials.Stone', 'one_hot_CommunityFeatures.Sidewalks', 'one_hot_FrontageType.Lakefront', 'one_hot_dock', 'one_hot_stream(s)', 'one_hot_AssociationAmenities.ShuffleboardCourt', 'one_hot_Appliances.WaterHeater', 'one_hot_AssociationAmenities.SportCourt', 'one_hot_backs_to_trees/woods', 'one_hot_FrontageType.Oceanfront', 'one_hot_ExteriorFeatures.Storage', 'one_hot_Heating.Electric', 'one_hot_ParkingFeatures.DetachedCarport', 'one_hot_Heating.Ceiling', 'one_hot_Basement.Unfinished', 'one_hot_WaterfrontFeatures.OceanFront', 'one_hot_spring(s)', 'one_hot_CommunityFeatures.StreetLights', 'one_hot_View.TreesWoods', 'one_hot_WaterfrontFeatures.Pond', 'one_hot_pond(s)', 'one_hot_Roof.Wood', 'one_hot_View.Lake', 'one_hot_sloped', 'one_hot_rear_of_lot', 'one_hot_adjoins_government_land', 'one_hot_LaundryFeatures.InKitchen', 'one_hot_View.Rural', 'one_hot_ExteriorFeatures.Barbecue', 'one_hot_ExteriorFeatures.Playground', 'one_hot_AssociationAmenities.FitnessCenter', 'one_hot_PoolFeatures.Community', 'one_hot_legal_non-conforming', 'one_hot_ExteriorFeatures.BuiltInBarbecue', 'one_hot_ExerciseRoom', 'one_hot_ExteriorFeatures.PrivateYard', 'one_hot_Roof.Slate', 'one_hot_chain_link_fence', 'one_hot_irregular_lot', 'one_hot_InteriorOrRoomFeatures.WetBar', 'one_hot_electric_fence', 'one_hot_DoorFeatures.FrenchDoors', 'one_hot_multi', 'one_hot_Balcony', 'one_hot_Appliances.Microwave', 'one_hot_Heating.Baseboard', 'one_hot_Appliances.Range', 'one_hot_FloorPlan', 'one_hot_AssociationAmenities.IndoorPool', 'one_hot_OtherStructures.Stables', 'one_hot_outdoor_lighting', 'one_hot_Attic', 'one_hot_Fence', 'one_hot_mature_trees', 'one_hot_Appliances.GasCooktop', 'one_hot_View', 'one_hot_Patio', 'one_hot_Bar', 'one_hot_sidewalks', 'one_hot_LaundryFeatures.InGarage', 'one_hot_water_rights', 'one_hot_InteriorOrRoomFeatures.BeamedCeilings', 'one_hot_FireplaceFeatures.DiningRoom', 'one_hot_Hallway', 'one_hot_AssociationAmenities.Pool', 'one_hot_corner_lot', 'one_hot_Appliances.RangeHood', 'one_hot_AerialView', 'one_hot_Appliances.Cooktop', 'one_hot_PatioAndPorchFeatures.Patio', 'one_hot_agric', 'one_hot_Flooring.Terrazzo', 'one_hot_InteriorOrRoomFeatures.CofferedCeilings', 'one_hot_Flooring.Brick', 'one_hot_SpaFeatures.Private', 'one_hot_LaundryFeatures.InBasement', 'one_hot_LaundryFeatures.LaundryRoom', 'one_hot_FireplaceFeatures.LivingRoom', 'one_hot_Appliances.Washer', 'one_hot_Appliances.WasherDryerStacked', 'one_hot_SecurityFeatures.SmokeDetectors', 'one_hot_Heating.Fireplaces', 'one_hot_Appliances.WasherDryer', 'one_hot_WalkInClosets', 'one_hot_Appliances.Refrigerator', 'one_hot_ExteriorFeatures.Dock', 'one_hot_AssociationAmenities.Elevators', 'one_hot_InteriorOrRoomFeatures.TrackLighting', 'one_hot_chain_of_lakes_frontage', 'one_hot_LaundryFeatures.LaundryCloset', 'one_hot_ExteriorFeatures.FirePit', 'one_hot_ExteriorFeatures.Awnings', 'one_hot_Appliances.DoubleOven', 'one_hot_View.Skyline', 'one_hot_FireplaceFeatures.Outside', 'one_hot_LaundryFeatures.Sink', 'one_hot_FireplaceFeatures.RaisedHearth', 'one_hot_InteriorOrRoomFeatures.Bar', 'one_hot_central_individual', 'one_hot_InteriorOrRoomFeatures.WalkInClosets', 'one_hot_Appliances.GasRange', 'one_hot_PoolFeatures.InGround', 'one_hot_Flooring.Hardwood', 'one_hot_ParkingFeatures.Attached', 'one_hot_InteriorOrRoomFeatures.CrownMolding', 'one_hot_landscaped', 'one_hot_Levels.Two', 'one_hot_Electric.Generator', 'one_hot_Gym', 'one_hot_WineCellar', 'one_hot_Appliances.Oven', 'one_hot_Other', 'one_hot_waterfront', 'one_hot_LaundryFeatures.ElectricDryerHookup', 'one_hot_Community', 'one_hot_Appliances.IceMaker', 'one_hot_Flooring.Carpet', 'one_hot_commercial_lease', 'one_hot_Stable', 'one_hot_farm', 'one_hot_WaterfrontFeatures.Creek', 'one_hot_OtherStructures.OutdoorKitchen', 'one_hot_park_adjacent', 'one_hot_water_view', 'one_hot_geothermal', 'one_hot_Kitchen', 'one_hot_InteriorOrRoomFeatures.StoneCounters', 'one_hot_View.Water', 'one_hot_Yard', 'one_hot_none', 'one_hot_ExteriorFeatures.OutdoorKitchen', 'one_hot_SecurityFeatures.FireSprinklerSystem', 'one_hot_InteriorOrRoomFeatures.BreakfastBar', 'one_hot_paddock', 'one_hot_InteriorOrRoomFeatures.DoubleVanity', 'one_hot_PatioAndPorchFeatures.FrontPorch', 'one_hot_singl', 'one_hot_Pool', 'one_hot_Appliances.Dishwasher', 'one_hot_lake_access', 'one_hot_WaterfrontFeatures.BeachFront', 'one_hot_PoolFeatures.Indoor', 'one_hot_SpaFeatures.InGround', 'one_hot_ParkingFeatures.Garage', 'one_hot_lake_front', 'ImageData.c1c6.summary.kitchen', 'one_hot_Appliances.BuiltInRefrigerator', 'one_hot_SunRoom', 'one_hot_Lobby', 'one_hot_zoned', 'one_hot_View.Pool', 'one_hot_InteriorOrRoomFeatures.BuiltInFeatures', 'one_hot_central_air', 'one_hot_Appliances.StainlessSteelAppliances', 'one_hot_Cooling.CeilingFans', 'one_hot_residential', 'Structure.NewConstructionYN', 'one_hot_space_pac', 'one_hot_MudRoom', 'day_sin', 'age_bracket', 'one_hot_manufactured_in_park', 'day_cos', 'one_hot_common_grounds', 'one_hot_commercial_sale', 'ImageData.style.stories.summary.label', 'one_hot_beach', 'one_hot_horses_allowed', 'month_sin', 'ImageData.c1c6.summary.exterior', 'ImageData.c1c6.summary.bathroom', 'one_hot_View.City', 'Structure.YearBuilt', 'month_cos', 'ImageData.c1c6.summary.property', 'ImageData.c1c6.summary.interior', 'below_grade_ratio', 'one_hot_residential_income', 'Structure.BedroomsTotal', 'bedroom_ratio', 'living_to_lot_ratio', 'ImageData.q1q6.summary.bathroom', 'ImageData.q1q6.summary.exterior', 'interior_quality', 'Structure.BelowGradeUnfinishedArea', 'Structure.Rooms.RoomsTotal', 'ImageData.q1q6.summary.interior', 'Structure.BathroomsHalf', 'Structure.BelowGradeFinishedArea', 'ImageData.q1q6.summary.kitchen', 'Structure.BathroomsFull', 'Structure.GarageSpaces', 'room_density', 'bathroom_density', 'Structure.FireplacesTotal', 'Structure.LivingArea', 'property_age', 'ImageData.q1q6.summary.property', 'Polar.Theta', 'Polar2', 'Characteristics.LotSizeSquareFeet', 'total_bathrooms', 'Listing.Price.ClosePrice', 'Listing.Dates.CloseDate']

        print(f'Columns input file: {df.iloc[0]}')
        
        # Convert date column to datetime
        df['Listing.Dates.CloseDate'] = pd.to_datetime(df['Listing.Dates.CloseDate'])

        X = df.drop(columns=['Listing.Price.ClosePrice'])

        y = df['Listing.Price.ClosePrice']

        print(f'Columns input file after transf: {len(X.iloc[0])}')

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
    


    def show_rellevant_features(self, data):
        categories = list(data.keys())
        values = list(data.values())

        # Sort data by values for better visualization
        sorted_indices = np.argsort(values)
        categories = [categories[i] for i in sorted_indices[-500:-1]]
        values = [values[i] for i in sorted_indices[-500:-1]]

    
        # Create the plot
        plt.figure(figsize=(10, len(categories) * 0.3))  # Adjust height for better fit
        plt.barh(categories, values, color="skyblue")
        plt.xlabel("Values")
        plt.title("Horizontal Bar Chart")
        plt.tight_layout()

        # Save the plot
        plot_path = "horizontal_barchart.png"
        plt.savefig(plot_path)
        plt.close()

        mlflow.log_artifact(plot_path)
        print(f"Bar chart logged as artifact: {plot_path}")


def create_engineered_features(df):
    df['Polar2'] = pow(df['Polar.Theta'], 2)
    df['Polar3'] = pow(df['Polar.R'], 4)

    # Area ratios
    df['living_to_lot_ratio'] = np.log( df['Structure.LivingArea'] / df['Characteristics.LotSizeSquareFeet'])
    df['below_grade_ratio'] = np.log(df['Structure.BelowGradeFinishedArea'] / df['Structure.LivingArea'])
    
    # Bathroom features
    df['total_bathrooms'] = df['Structure.BathroomsFull'] + 0.5 * df['Structure.BathroomsHalf']
    df['bathroom_density'] = df['total_bathrooms'] / df['Structure.LivingArea']
    
    # Age features
    current_year = 2024
    df['property_age'] = current_year - df['Structure.YearBuilt']
    df['age_bracket'] = pd.cut(df['property_age'], 
                            bins=[0, 10, 20, 30, 40, 50, float('inf')],
                            labels=['0-10', '11-20', '21-30', '31-40', '41-50', '50+'])
    
    # Room metrics
    df['room_density'] = df['Structure.Rooms.RoomsTotal'] / df['Structure.LivingArea']
    df['bedroom_ratio'] = df['Structure.BedroomsTotal'] / df['Structure.Rooms.RoomsTotal']
    
    # Quality scores
    df['interior_quality'] = df[[col for col in df.columns if 'interior' in col.lower()]].mean(axis=1)

    return df



def generate_predictions_file(
    model,                    # Your trained LightGBM model
    validation_df,           # Validation DataFrame without target
    lookup_df,
    id_column='Listing.ListingId', # Name of your ID column
    output_path='predictions.csv',  # Where to save the predictions
    feature_columns=None, 
    ):
    try:
        # Make a copy to avoid modifying the original dataframe
        val_df = validation_df.copy()
        val_df = create_engineered_features(val_df)
        val_df = val_df.drop(columns=['Listing.Price.ClosePrice', 'Listing.Dates.CloseDate'])
        
        # Generate predictions
        predictions = model.predict(val_df)
        
        # Create output dataframe
        output_df = pd.DataFrame({
            'ID': lookup_df[id_column],
            'predicted_price': predictions
        })
        
        # Sort by ID
        output_df = output_df.sort_values('ID')
        
        # Save to CSV
        output_df.to_csv(output_path, index=False)
        
        print(f"Predictions saved to {output_path}")
        print(f"Number of predictions: {len(predictions)}")
        print("\nPreview of predictions:")
        print(output_df.head())
        print("\nSummary statistics:")
        print(output_df['predicted_price'].describe())
        
        return output_df
        
    except Exception as e:
        print(f"Error generating predictions: {str(e)}")
        raise


def get_ensemble_configs():
    # Model 1: Focused on deep relationships with more trees and depth
    deep_model_params = {
        "n_estimators": 200,
        "max_depth": 12,
        "num_leaves": 100,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "min_split_gain": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1
    }
    
    # Model 2: Focused on preventing overfitting with regularization
    robust_model_params = {
        "n_estimators": 150,
        "max_depth": 8,
        "num_leaves": 50,
        "learning_rate": 0.1,
        "min_child_samples": 30,
        "min_split_gain": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "random_state": 43,
        "verbose": -1
    }
    
    # Model 3: Focused on local patterns with smaller trees but more of them
    local_model_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "num_leaves": 32,
        "learning_rate": 0.03,
        "min_child_samples": 10,
        "min_split_gain": 0.01,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 44,
        "verbose": -1
    }
    
    return [deep_model_params, robust_model_params, local_model_params]

def get_ensemble_weights():
    return [0.4, 0.3, 0.3]  # Giving slightly more weight to the deep model


def main():
    # Clear existing mlruns directory if it exists (optional, remove if you want to keep history)

    mlruns_dir = "mlruns"
    if not os.path.exists(mlruns_dir):
        os.makedirs(mlruns_dir)

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), 'dataset'))
    train_file = os.path.abspath(os.path.join(parent_dir, 'df_train.csv'))
    
    model_configs = get_ensemble_configs()
    weights = get_ensemble_weights()
    model = EnsembleLightGBMModel(model_configs, weights)

    # Initialize predictor with experiment name
    predictor = HousePricePredictor(model, "house_price_experiment")
    predictor.cv = TimeSeriesCV(min_training_months=3, forecast_months=1)
    
    # Print starting message
    print("\nStarting House Price Prediction with Ensemble Model")
    print("="*50)
    print("Ensemble Configuration:")
    for i, config in enumerate(model_configs, 1):
        print(f"\nModel {i} Parameters (Weight: {weights[i-1]:.2f}):")
        for param, value in config.items():
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

    predictor.show_rellevant_features(model.get_feature_importance())

    print("\nMLflow UI can be started with:")
    print("mlflow ui --backend-store-uri file:./mlruns")
    validation_df = pd.read_csv('./dataset/df_test.csv')
    id_df = pd.read_csv('./dataset/test_modified.csv', low_memory=False)

    generate_predictions_file(model, validation_df, id_df)

if __name__ == "__main__":
    main()