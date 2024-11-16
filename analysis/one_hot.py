import pandas as pd
import numpy as np
import os
from pathlib import Path
import ast

def convert_columns_to_int(df):
    for col in df.columns:
        try:
            # Try to convert the column to integers
            df[col] = df[col].astype(np.int64)
        except ValueError:
            try:
                df[col] = df[col].astype(np.float32)
            except:
                # Not possible to convert
                pass
    return df

def string_to_list(input_string):
    if isinstance(input_string, str):
        try:
            result = ast.literal_eval(input_string)
            return result
        except (ValueError, SyntaxError):
            return []  # Return empty list if parsing fails
    return input_string  # Return original value if not a string

def one_hot_from_list(df, column_name):
    df[column_name] = df[column_name].apply(lambda x: x if isinstance(x, list) else [])
    unique_elements = set(element for lst in df[column_name] for element in lst)
    for element in unique_elements:
        one_hot_col_name = f"one_hot_{element}"
        df[one_hot_col_name] = df[column_name].apply(lambda lst: 1 if element in lst else 0)
    return df

def get_column_types(df):
    return df.dtypes

def get_unique_types(df):
    return set(df.dtypes)

def save_dataset(df, filename):
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")

# Main logic
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'dataset'))
test_file = os.path.abspath(os.path.join(parent_dir, 'test.csv'))
train_file = os.path.abspath(os.path.join(parent_dir, 'train.csv'))

df_test = pd.read_csv(test_file, sep=',', low_memory=False)
df_train = pd.read_csv(train_file, sep=',', low_memory=False)

dataframes = [(df_train, 'df_train'), (df_test, 'df_test')]
columns_to_keep = ['Location.GIS.Longitude', 'Location.GIS.Latitude']
columns_to_one_hot = ["Characteristics.LotFeatures"]

for df, df_name in dataframes:
    # Drop unnecessary columns
    location_columns = [col for col in df.columns if col.startswith('Location')]
    columns_to_drop = [col for col in location_columns if col not in columns_to_keep]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Convert columns to lists and apply one-hot encoding
    for col in columns_to_one_hot:
        if col in df.columns:
            df[col] = df[col].apply(string_to_list)
            df = one_hot_from_list(df, col)

    # Save the updated DataFrame
    output_path = os.path.abspath(os.path.join(parent_dir, f'{df_name}.csv'))
    save_dataset(df, output_path)

    print(f"Processed columns for {df_name}: {list(df.columns)}")
