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
    df[column_name] = df[column_name].apply(lambda x: x if isinstance(x, list) else [])
    unique_elements = set(element for lst in df[column_name] for element in lst)
    for element in unique_elements:
        one_hot_col_name = f"one_hot_{element}"
        df[one_hot_col_name] = df[column_name].apply(lambda lst: 1 if element in lst else 0)
    return df

def save_dataset(df, filename):
    """Saves the DataFrame to a CSV file."""
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")

def preprocess_dataframe(df, config):
    # Apply one-hot encoding
    for col in config['columns_to_one_hot']:
        if col in df.columns:
            df[col] = df[col].apply(string_to_list)
            df = one_hot_from_list(df, col)
        df.drop(columns=col, inplace=True, errors='ignore')
    return df

def main():
    """Main execution function."""
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'dataset'))
    test_file = os.path.abspath(os.path.join(parent_dir, 'df_test.csv'))
    train_file = os.path.abspath(os.path.join(parent_dir, 'df_train.csv'))

    df_test = pd.read_csv(test_file, sep=',', low_memory=False)
    df_train = pd.read_csv(train_file, sep=',', low_memory=False)

    dataframes = [(df_train, 'df_train'), (df_test, 'df_test')]

    config = {
        'columns_to_one_hot': ["Characteristics.LotFeatures"]
    }

    for df, df_name in dataframes:
        df = preprocess_dataframe(df, config)
        output_path = os.path.abspath(os.path.join(parent_dir, f'{df_name}.csv'))
        save_dataset(df, output_path)

if __name__ == "__main__":
    main()
