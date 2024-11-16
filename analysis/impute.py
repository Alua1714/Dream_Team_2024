import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.preprocessing import LabelEncoder

class LightGBMImputer:
    def __init__(self, catsegorical_features=None, n_estimators=100, num_leaves=31, random_state=42):
        """
        Initialize the imputer.
        
        Parameters:
        -----------
        categorical_features : list, optional
            List of categorical column names
        n_estimators : int, default=100
            Number of trees in the random forest
        num_leaves : int, default=31
            Maximum number of leaves in each tree
        random_state : int, default=42
            Random state for reproducibility
        """
        self.categorical_features = categorical_features or []
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.models = {}
        self.label_encoders = {}
        
    def _prepare_features(self, X, target_column):
        """Prepare features by handling categorical variables and removing target column."""
        X_prep = X.copy()
        
        # Encode categorical variables
        for cat_col in self.categorical_features:
            if cat_col not in self.label_encoders:
                self.label_encoders[cat_col] = LabelEncoder()
                # Fill missing values with a placeholder before encoding
                X_prep[cat_col].fillna('MISSING', inplace=True)
                self.label_encoders[cat_col].fit(X_prep[cat_col])
            X_prep[cat_col] = self.label_encoders[cat_col].transform(X_prep[cat_col])
        
        # Remove target column from features
        features = X_prep.drop(columns=[target_column])
        return features
        
    def fit(self, X, columns_to_impute):
        """
        Fit imputation models for specified columns.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
        columns_to_impute : list
            List of column names to impute
        """
        self.columns_to_impute = columns_to_impute
        
        for column in columns_to_impute:
            # Get rows where target column is not null
            train_mask = ~X[column].isna()
            if train_mask.sum() == 0:
                raise ValueError(f"No non-null values in column {column} to train on")
            
            # Prepare features
            X_train = self._prepare_features(X[train_mask], column)
            y_train = X[column][train_mask]
            
            # Choose model type based on whether column is categorical
            if column in self.categorical_features:
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                    y_train = self.label_encoders[column].fit_transform(y_train)
                model = LGBMClassifier(
                    n_estimators=self.n_estimators,
                    num_leaves=self.num_leaves,
                    random_state=self.random_state
                )
            else:
                model = LGBMRegressor(
                    n_estimators=self.n_estimators,
                    num_leaves=self.num_leaves,
                    random_state=self.random_state
                )
            
            # Fit model
            model.fit(X_train, y_train)
            self.models[column] = model
            
    def transform(self, X):
        """
        Impute missing values in specified columns.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with missing values
            
        Returns:
        --------
        pandas.DataFrame
            Data with imputed values
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
                predictions = self.models[column].predict(X_predict)
                
                # If categorical, decode predictions
                if column in self.categorical_features:
                    predictions = self.label_encoders[column].inverse_transform(predictions)
                
                # Fill missing values
                X_imputed.loc[impute_mask, column] = predictions
                
        return X_imputed
    
    def fit_transform(self, X, columns_to_impute):
        """
        Fit models and impute missing values.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with missing values
        columns_to_impute : list
            List of column names to impute
            
        Returns:
        --------
        pandas.DataFrame
            Data with imputed values
        """
        self.fit(X, columns_to_impute)
        return self.transform(X)