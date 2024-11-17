import pandas as pd
import numpy as np
import os
import ast
from pathlib import Path

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

def main():
    """Main execution function."""
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'dataset'))
    test_file = os.path.abspath(os.path.join(parent_dir, 'test.csv'))
    train_file = os.path.abspath(os.path.join(parent_dir, 'train.csv'))

    df_test = pd.read_csv(test_file, sep=',', low_memory=False)
    df_train = pd.read_csv(train_file, sep=',', low_memory=False)

    dataframes = [(df_train, 'df_del_train'), (df_test, 'df_del_test')]

    config = {
        'columns_to_drop': [
            "ImageData.features_reso.results", "ImageData.room_type_reso.results",
            "ImageData.style.exterior.summary.label", "Structure.Basement",
            "Structure.Cooling", "Structure.Heating", "Structure.ParkingFeatures",
            "UnitTypes.UnitTypeType", "Listing.ListingId", "Property.PropertyType",
            "Tax.Zoning",
        ],
        'date_columns': ["Listing.Dates.CloseDate"],
        'boolean_columns': ["Structure.NewConstructionYN"],
        'float_conversion_columns': ["ImageData.style.stories.summary.label"],
        'columns_to_one_hot': ["Characteristics.LotFeatures"]
    }

    for df, df_name in dataframes:
        df = preprocess_dataframe(df, config)
        output_path = os.path.abspath(os.path.join(parent_dir, f'{df_name}.csv'))
        save_dataset(df, output_path)

if __name__ == "__main__":
    main()
