import pandas as pd
import numpy as np
import os
import ast
from pathlib import Path

def string_to_list(input_string):
    """Converts a string representation of a list into an actual list."""
    if isinstance(input_string, str):
        try:
            return ast.literal_eval(input_string)
        except (ValueError, SyntaxError):
            return []
    return input_string

def one_hot_from_list(df, column_name):
    """Creates one-hot encoding for elements in a column of lists."""
    # Ensure all values in the column are lists
    df[column_name] = df[column_name].apply(lambda x: x if isinstance(x, list) else [])
    
    # Extract unique elements across all lists
    unique_elements = set(element for lst in df[column_name] for element in lst)
    
    # Create one-hot encoded columns
    for element in unique_elements:
        one_hot_col_name = f"one_hot_{element}"
        df[one_hot_col_name] = df[column_name].apply(lambda lst: 1 if element in lst else 0)
    
    return df

def save_dataset(df, filename):
    """Saves the DataFrame to a CSV file."""
    try:
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
    except Exception as e:
        print(f"Error saving dataset to {filename}: {e}")

def preprocess_dataframe(df, config):
    """Preprocess the DataFrame by applying one-hot encoding."""
    for col in config['columns_to_one_hot']:
        print(df.columns)

        if col in df.columns:
            # Convert string representations of lists to actual lists
            df[col] = df[col].apply(string_to_list)
            # Apply one-hot encoding
            df = one_hot_from_list(df, col)
            # Drop the original column
            df.drop(columns=col, inplace=True, errors='ignore')
        else:
            print(f"Column '{col}' not found in DataFrame. Skipping...")
    return df

def main():
    """Main execution function."""
    try:
        # Define file paths
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'dataset'))
        test_file = os.path.join(parent_dir, 'test_imputed.csv')
        train_file = os.path.join(parent_dir, 'train_imputed.csv')

        # Load datasets
        df_test = pd.read_csv(test_file, sep=',', low_memory=False)
        df_train = pd.read_csv(train_file, sep=',', low_memory=False)

        # Config for preprocessing
        config = {
            'columns_to_one_hot': ["Characteristics.LotFeatures"]
        }

        # Preprocess and save datasets
        df_train = preprocess_dataframe(df_train, config)
        save_dataset(df_train, os.path.join(parent_dir, 'df_train.csv'))
        
        df_test = preprocess_dataframe(df_test, config)
        save_dataset(df_test, os.path.join(parent_dir, 'df_test.csv'))

    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
