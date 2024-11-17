import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
import ast
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import nan_euclidean_distances
import pickle


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
    

# config2ure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_paths():
    """Setup and return all necessary paths."""
    # Get directory containing current script
    script_dir = Path(__file__).parent
    modified_dir = script_dir / '../dataset'
    modified_dir.mkdir(exist_ok=True)
    
    return {
        'test': modified_dir / 'test_modified.csv',
        'train': modified_dir / 'df_del_train.csv',
        'modified': script_dir / 'data'
    }

important_locations = [
    (41.82064698842478, -87.79964386423387),
    (41.902418293457885, -87.60154275379254),
    (41.87897380756125, -87.62827578338285),
    (41.910428444271915, -87.70059401735486),
    (41.92691533524087, -87.69628146100543),
    (41.8680506170819, -87.61805717359894),
    (41.63641020573378, -88.53540916403767),
    (41.9290179188145, -87.63405761365621),
    (42.00268512389682, -87.91126405667943),
    (41.80235064745896, -87.75745547017502),
    (42.12808414490979, -87.90027774207515),
    (42.07555182606426, -87.6948477743019),
    (41.867930975959034, -87.62447480780993),
    (41.644610141763216, -87.5127716767169)
]

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process a DataFrame to add missing latitude and longitude data."""
    df = df.copy()
    
    # Create mask for confidential locations (that are fully redacted)
    mask = (df["Location.GIS.Latitude"] == 40.6331249) & (df["Location.GIS.Longitude"] == -89.3985283)
    df.loc[mask, ["Location.GIS.Latitude", "Location.GIS.Longitude"]] = pd.NA

    # Manually fill rows that are not correct
    shelbyville_mask = df["Location.Address.UnparsedAddress"].str.lower() == "407 sw fifth street, shelbyville, il 62565"
    normal_mask = df["Location.Address.UnparsedAddress"].str.lower() == "712 & 715 golfcrest road, normal, il 61761"
    
    # Update coordinates for Shelbyville address
    df.loc[shelbyville_mask, "Location.GIS.Latitude"] = 39.402713789610495
    df.loc[shelbyville_mask, "Location.GIS.Longitude"] = -88.79728330743421
    # Update coordinates for Normal address
    df.loc[normal_mask, "Location.GIS.Latitude"] = 40.529155234759784
    df.loc[normal_mask, "Location.GIS.Longitude"] = -89.00127842461578
    
    return cartesian_to_polar(df)


def cartesian_to_polar(df: pd.DataFrame) -> pd.DataFrame:
    """Convert latitude/longitude to polar coordinates (r, theta)."""
    df = df.copy()
    
    CENTER_LAT = 41.87698087663472
    CENTER_LON = -87.63402335655448
    R = 6371000

    # Initialize columns with NaN
    df["Polar.R"] = np.nan
    df["Polar.Theta"] = np.nan

    # Get mask for valid coordinates
    valid_mask = df["Location.GIS.Latitude"].notna() & df["Location.GIS.Longitude"].notna()
    
    # Only process valid coordinates
    if valid_mask.any():
        lat = np.radians(df.loc[valid_mask, "Location.GIS.Latitude"])
        lon = np.radians(df.loc[valid_mask, "Location.GIS.Longitude"])
        center_lat = np.radians(CENTER_LAT)
        center_lon = np.radians(CENTER_LON)
        
        # Calculate distance (r) using Haversine formula
        dlat = lat - center_lat
        dlon = lon - center_lon
        x = R * dlon * np.cos((lat + center_lat) / 2)  # Adjust for latitude distortion
        y = R * dlat

        # Calculate bearing (θ)
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        # Assign results only to valid coordinates
        df.loc[valid_mask, "Polar.R"] = r
        df.loc[valid_mask, "Polar.Theta"] = theta
        
        for i, (lat, lng) in enumerate(important_locations):
            lat, lng = np.radians(lat), np.radians(lng)
            dlat = lat - center_lat
            dlon = lon - center_lon
            xd = R * dlon * np.cos((lat + center_lat) / 2)  # Adjust for latitude distortion
            yd = R * dlat

            df.loc[valid_mask, f"Distance.{i}"] = np.sqrt((x - xd)**2 + (y - yd)**2)
    columns_to_drop = [col for col in df.columns if col.startswith('Location')]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    return df

def convert_columns_to_int(df):
    """Converts columns to integers if possible, otherwise tries to convert to float."""
    for col in df.columns:
        try:
            df[col] = df[col].astype(np.int64)
        except ValueError:
            try:
                df[col] = df[col].astype(np.float32)
            except Exception:
                pass
    return df

def string_to_list(input_string):
    """Converts a string representation of a list into an actual list."""
    if isinstance(input_string, str):
        try:
            return ast.literal_eval(input_string)
        except (ValueError, SyntaxError):
            return []
    return input_string

def save_dataset(df, filename):
    """Saves the DataFrame to a CSV file."""
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")

def preprocess_dataframe(df, config):
    """Applies a series of preprocessing steps to the DataFrame."""
    # Drop unnecessary columns
    col_drop = [col for col in config['columns_to_drop'] if col not in config["columns_to_one_hot"]]
    df.drop(columns=col_drop, inplace=True, errors='ignore')

    # Convert date columns
    for col in config['date_columns']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%Y-%m-%dT%H:%M:%S", errors='coerce')

    # Convert boolean columns
    for col in config['boolean_columns']:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # Convert specific columns to float
    for col in config['float_conversion_columns']:
        if col in df.columns:
            df[col] = df[col].apply(extract_and_convert_to_float)

    return df

def extract_and_convert_to_float(input_string):
    """Extracts a substring and converts it to float."""
    try:
        substring = input_string.split('_')[0]
        return np.float32(substring)
    except (ValueError, AttributeError):
        return None

def string_to_list(input_string):
    """Converts a string representation of a list into an actual list."""
    if isinstance(input_string, str):
        try:
            return ast.literal_eval(input_string)
        except (ValueError, SyntaxError):
            return []
    return input_string


def string_list_2(value):
    if pd.isna(value):
        return np.nan
    elif isinstance(value, str):
        return [value]
    else:
        return value
    

def one_hot_from_list(df, column_name, unique_elements):
    """Creates one-hot encoding for elements in a column of lists."""
    # Ensure all values in the column are lists
    df[column_name] = df[column_name].apply(lambda x: x if isinstance(x, list) else [])
    # Extract unique elements across all lists
    for element in unique_elements:
        new_el = element.replace(" ", "_")
        one_hot_col_name = f"one_hot_{new_el}"
        df[one_hot_col_name] = df[column_name].apply(lambda lst: 1 if element in lst else 0)
    print(len(df), "AFTER ONE HOT")
    return df

def save_dataset(df, filename):
    """Saves the DataFrame to a CSV file."""
    try:
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
    except Exception as e:
        print(f"Error saving dataset to {filename}: {e}")

def preprocess_dataframe2(df, config, uniques):
    """Preprocess the DataFrame by applying one-hot encoding."""
    for col in config['prepare']:
        df[col] = df[col].apply(string_list_2)

    for col in config['columns_to_one_hot']:
        if col in df.columns:
            # Convert string representations of lists to actual lists
            df[col] = df[col].apply(string_to_list)
            # Apply one-hot encoding
            df = one_hot_from_list(df, col,uniques[col])
            # Drop the original column
            df.drop(columns=col, inplace=True, errors='ignore')
        else:
            print(f"Column '{col}' not found in DataFrame. Skipping...")
    return df

def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data
def load_dictionary(path):

    try:
        with open(path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise Exception(f"An error occurred while loading the dictionary: {e}")
    
def process_data(df_test: pd.DataFrame) -> pd.DataFrame:
    # Setup paths

    paths = setup_paths()

    try:
        # Read and process datasets
        logging.info("Processing test dataset...")
        df_test = clean_dataframe(df_test)
        
        logging.info("Processing train dataset...")
        df_train = pd.read_csv(paths['train'], low_memory=False)
        
        logging.info("Modified datasets have been saved successfully.")

        config = {
            'columns_to_drop': [
                "ImageData.features_reso.results", "ImageData.room_type_reso.results",
                "ImageData.style.exterior.summary.label", "Structure.Basement",
                "Structure.Cooling", "Structure.Heating", "Structure.ParkingFeatures",
                "UnitTypes.UnitTypeType", "Listing.ListingId", "Property.PropertyType",
                "Tax.Zoning","ImageData.style.exterior.summary.label",
            ],
            'date_columns': ["Listing.Dates.CloseDate"],
            'boolean_columns': ["Structure.NewConstructionYN"],
            'float_conversion_columns': ["ImageData.style.stories.summary.label"],
            'columns_to_one_hot': ["Characteristics.LotFeatures","Structure.Cooling","Tax.Zoning","Property.PropertyType","ImageData.features_reso.results","ImageData.room_type_reso.results"]
        }

        df_test = preprocess_dataframe(df_test, config)
        
        ##IMPUTE VALUES
        train_size = len(df_train)
        test_size = len(df_test)

        df_combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)
        drop_but_safe = ["Listing.Dates.CloseDate","Characteristics.LotFeatures",
                        "Structure.Cooling","Tax.Zoning","Property.PropertyType",
                        "ImageData.features_reso.results","ImageData.room_type_reso.results"]

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
        df_train= imputed_data.iloc[:train_size, :].reset_index(drop=True)
        df_test = imputed_data.iloc[train_size:, :].reset_index(drop=True)

        #ONE HOT
        uniques = load_dictionary(os.path.join(paths['modified'],"saved_data.pkl"))
        train_size = len(df_train)
        # Config for preprocessing
        config2 = {
            'prepare':["Tax.Zoning","Property.PropertyType"],
            'columns_to_one_hot': ["Characteristics.LotFeatures","Structure.Cooling","Tax.Zoning","Property.PropertyType","ImageData.features_reso.results","ImageData.room_type_reso.results"]
        }

        # Ensure 'month' is the numeric month value
        df_test = preprocess_dataframe2(df_test,config2,uniques)
        #AQUí ENCARA SENCER
            
        df_test['Listing.Dates.CloseDate'] = pd.to_datetime(df_test['Listing.Dates.CloseDate'], errors='coerce')

    # Extract month and encode it
        df_test['month'] = df_test['Listing.Dates.CloseDate'].dt.month
        df_test = encode(df_test, 'month', 12)

        # Extract day and encode it
        df_test['day'] = df_test['Listing.Dates.CloseDate'].dt.day
        df_test = encode(df_test, 'day', 31)
        df_test.drop(columns=['month','day'],inplace=True, errors='ignore')

        df_test.drop(["Listing.Price.ClosePrice", "Listing.Dates.CloseDate"], axis=1, inplace=True)
        df_test.to_csv('data/processed.csv', index=False)
        return df_test
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise