import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import nan_euclidean_distances
import os

class HybridImputer:
    """
    A class to impute missing values using LightGBM random forest for numerical columns
    and k-Nearest Neighbors for categorical columns.
    """
    
    def __init__(self, categorical_features=None, n_estimators=100, num_leaves=31, 
                 n_neighbors=5, random_state=42):
        """
        Initialize the imputer.
        """
        self.categorical_features = categorical_features or []
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.models = {}
        self.label_encoders = {}
        
    def _fit_label_encoders(self, X):
        """Fit label encoders on all data first to capture all categories."""
        for cat_col in self.categorical_features:
            if cat_col not in self.label_encoders:
                self.label_encoders[cat_col] = LabelEncoder()
                # Get all unique values including 'MISSING' for NaN
                unique_values = pd.concat([
                    pd.Series(['MISSING']),
                    X[cat_col].dropna()
                ]).unique()
                self.label_encoders[cat_col].fit(unique_values)
    
    def _prepare_features(self, X, target_column):
        """Prepare features by handling categorical variables and removing target column."""
        X_prep = X.copy()
        
        # Handle categorical features
        for cat_col in self.categorical_features:
            # Replace NaN with 'MISSING'
            X_prep[cat_col] = X_prep[cat_col].fillna('MISSING')
            X_prep[cat_col] = self.label_encoders[cat_col].transform(X_prep[cat_col])
        
        # Remove target column from features
        features = X_prep.drop(columns=[target_column])
        
        # Convert to numeric type for distance calculation
        features = features.astype(float)
        
        return features
    
    def _get_mode_from_neighbors(self, X_train, y_train, X_predict, k):
        """Get mode of k nearest neighbors using nan_euclidean_distances."""
        # Calculate distances considering NaN values
        distances = nan_euclidean_distances(X_predict, X_train)
        
        # Get indices of k nearest neighbors for each prediction point
        neighbor_indices = np.argpartition(distances, k, axis=1)[:, :k]
        
        # Get modes of neighbor values
        predictions = []
        for indices in neighbor_indices:
            neighbor_values = y_train.iloc[indices]
            predictions.append(neighbor_values.mode().iloc[0])
            
        return np.array(predictions)
        
    def fit(self, X, columns_to_impute):
        """
        Fit imputation models for specified columns.
        """
        self.columns_to_impute = columns_to_impute
        X = X.copy()
        
        # Fit label encoders on all data first
        self._fit_label_encoders(X)
        
        for column in columns_to_impute:
            # Get rows where target column is not null
            train_mask = ~X[column].isna()
            if train_mask.sum() == 0:
                raise ValueError(f"No non-null values in column {column} to train on")
            
            # Prepare features
            X_train = self._prepare_features(X[train_mask], column)
            y_train = X[column][train_mask]
            
            if column in self.categorical_features:
                # For categorical columns, store training data for KNN
                self.models[column] = {
                    'X_train': X_train,
                    'y_train': y_train
                }
            else:
                # For numerical columns, use LightGBM with adjusted parameters
                model = LGBMRegressor(
                    n_estimators=self.n_estimators,
                    num_leaves=self.num_leaves,
                    random_state=self.random_state,
                    min_data_in_leaf=1,
                    min_data_in_bin=1
                )
                model.fit(X_train, y_train)
                self.models[column] = model
            
    def transform(self, X):
        """
        Impute missing values in specified columns.
        """
        X_imputed = X.copy()
        
        for column in self.columns_to_impute:
            # Only impute if there are missing values
            if X_imputed[column].isna().sum() > 0:
                # Get rows where imputation is needed
                impute_mask = X_imputed[column].isna()
                
                # Prepare features
                X_predict = self._prepare_features(X_imputed[impute_mask], column)
                
                # Predict missing values
                if column in self.categorical_features:
                    # Use KNN for categorical columns
                    predictions = self._get_mode_from_neighbors(
                        self.models[column]['X_train'],
                        self.models[column]['y_train'],
                        X_predict,
                        self.n_neighbors
                    )
                else:
                    # Use LightGBM for numerical columns
                    predictions = self.models[column].predict(X_predict)
                
                # Fill missing values
                X_imputed.loc[impute_mask, column] = predictions
                
        return X_imputed
    
    def fit_transform(self, X, columns_to_impute):
        """
        Fit models and impute missing values.
        """
        self.fit(X, columns_to_impute)
        return self.transform(X)
    

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'dataset'))
test_file = os.path.abspath(os.path.join(parent_dir, 'df_del_test.csv'))
train_file = os.path.abspath(os.path.join(parent_dir, 'df_del_train.csv'))

df_test = pd.read_csv(test_file, sep=',', low_memory=False)
df_train = pd.read_csv(train_file, sep=',', low_memory=False)
train_size = len(df_train)
test_size = len(df_test)

df_combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)
drop_but_safe = ["Listing.Dates.CloseDate","Characteristics.LotFeatures",
                 "Structure.Cooling","Tax.Zoning","Property.PropertyType"]

safed_cols = df_combined[drop_but_safe]
df_combined.drop(columns=drop_but_safe, inplace=True, errors='ignore')


#df_combined.to_csv('df_combined.csv', index=False)

imputer = HybridImputer(
    categorical_features=None,
    n_estimators=100,
    n_neighbors=3
)

# Impute missing values
cols_impute = list(df_combined.columns)
imputed_data = imputer.fit_transform(df_combined, columns_to_impute=cols_impute)
imputed_data[drop_but_safe] = safed_cols
print
df_train_recovered = imputed_data.iloc[:train_size, :].reset_index(drop=True)
df_test_recovered = imputed_data.iloc[train_size:, :].reset_index(drop=True)

output_path_train = os.path.abspath(os.path.join(parent_dir, f'train_imputed.csv'))
output_path_test = os.path.abspath(os.path.join(parent_dir, f'test_imputed.csv'))

df_train_recovered.to_csv(output_path_train, index=False)
df_test_recovered.to_csv(output_path_test, index=False)