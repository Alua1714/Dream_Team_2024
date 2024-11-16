import pandas as pd
import numpy as np
import os
from pathlib import Path
import ast

# Get the parent directory of the current working directory and append 'dataset'
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'dataset'))

# Construct the paths for 'test.csv' and 'train.csv'
test_file = os.path.abspath(os.path.join(parent_dir, 'test.csv'))
train_file = os.path.abspath(os.path.join(parent_dir, 'train.csv'))

df_test = pd.read_csv(test_file, sep=',', low_memory=False)
df_train = pd.read_csv(train_file, sep=',', low_memory=False)


dataframes = [(df_train, 'df_train'), (df_test, 'df_test')]

def convert_columns_to_int(df):
    for col in df.columns:
        try:
            # Try to convert the column to integers
            df[col] = df[col].astype(np.int64)
        except ValueError:
            try:
                df[col] = df[col].astype(np.float32)
            except:
                #not possible:
                pass
            pass
    return df

for df, df_name in dataframes:
    columns_to_keep = ['Location.GIS.Longitude', 'Location.GIS.Latitude']
    location_columns = [col for col in df.columns if col.startswith('Location')]
    columns_to_drop = [col for col in location_columns if col not in columns_to_keep]
    df = df.drop(columns=columns_to_drop)

# Function to convert string representations to actual lists
def string_to_list(input_string):
    if isinstance(input_string, str):  # Only process strings
        try:
            result = ast.literal_eval(input_string)
            return result
        except (ValueError, SyntaxError):
            return []  # Return empty list if parsing fails
    return input_string  # Return original value if not a string

# Function to create one-hot encoding from a column of lists
def one_hot_from_list(df, column_name):
    # Replace NaN or invalid entries with empty lists
    df[column_name] = df[column_name].apply(lambda x: x if isinstance(x, list) else [])
    
    # Get all unique elements from the lists in the specified column
    unique_elements = set(element for lst in df[column_name] for element in lst)
    # Iterate through each unique element and create a one-hot encoded column
    for element in unique_elements:
        one_hot_col_name = f"one_hot_{element}"
        df[one_hot_col_name] = df[column_name].apply(lambda lst: 1 if element in lst else 0)

    return df

# Example Usage
columns_to_one_hot = ["Characteristics.LotFeatures"]
rest_not_used = ["ImageData.features_reso.results","ImageData.room_type_reso.results","Structure.Cooling","Structure.Heating",
                       "Structure.ParkingFeatures"]
                     

for df, df_name in dataframes:
    for col in columns_to_one_hot:
        df[col] = df[col].apply(string_to_list)  # Convert strings to lists
        df = one_hot_from_list(df, col)  # Apply one-hot encoding

def get_column_types(df):
    return df.dtypes

def get_unique_types(df):
    return set(df.dtypes)

def save_dataset(df, filename):
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")
    
for df, df_name in dataframes:
    print(len(df.columns.tolist()))
    #print(get_column_types(df))
    print(get_unique_types(df))
    #print(df["ImageData.style.stories.summary.label"])
    p = os.path.abspath(os.path.join(parent_dir, f'{df_name}.csv'))
    save_dataset(df, p)
