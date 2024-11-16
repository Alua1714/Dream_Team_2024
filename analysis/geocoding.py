import pandas as pd
import os
import requests
from dotenv import load_dotenv
from pathlib import Path
import logging

def setup_logging():
    """Configure logging to both console and file."""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler('geocoding_logs.txt')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def setup_paths():
    """Setup and return all necessary paths."""
    base_dir = Path.cwd().parent
    dataset_dir = base_dir / 'dataset'
    modified_dir = dataset_dir / 'modified'
    modified_dir.mkdir(exist_ok=True)
    
    return {
        'test': dataset_dir / 'test.csv',
        'train': dataset_dir / 'train.csv',
        'modified': modified_dir
    }

def get_lat_lng(address: str, api_key: str) -> tuple[float | None, float | None]:
    """Get latitude and longitude for a given address using Google Maps Geocoding API."""
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": api_key}
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] == 'OK':
            location = data['results'][0]['geometry']['location']
            logging.info(f"Searching for: {address}")
            logging.info(f"lat:{location['lat']} lng:{location['lng']}")
            return location['lat'], location['lng']
        
        logging.warning(f"No results found for address: {address}")
        return None, None
            
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed for address {address}: {str(e)}")
        return None, None

def process_dataframe(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """Process a DataFrame to add missing latitude and longitude data."""
    df = df.copy()
    missing_coords = df[
        df["Location.GIS.Latitude"].isnull() | 
        df["Location.GIS.Longitude"].isnull()
    ]

    print(len(missing_coords))
    
    for index, row in missing_coords.iterrows():
        address = row["Location.Address.UnparsedAddress"]
        latitude, longitude = get_lat_lng(address, api_key)
        if latitude is not None and longitude is not None:
            df.at[index, "Location.GIS.Latitude"] = latitude
            df.at[index, "Location.GIS.Longitude"] = longitude
        else:
            logging.warning(f"Could not retrieve coordinates for address: {address}")
            
    return df

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY not found in environment variables")

    # Setup paths
    paths = setup_paths()
    
    try:
        # Read and process datasets
        logging.info("Processing test dataset...")
        df_test = pd.read_csv(paths['test'], low_memory=False)
        df_test_processed = process_dataframe(df_test, api_key)
        df_test_processed.to_csv(paths['modified'] / 'test_modified.csv', index=False)
        
        logging.info("Processing train dataset...")
        df_train = pd.read_csv(paths['train'], low_memory=False)
        df_train_processed = process_dataframe(df_train, api_key)
        df_train_processed.to_csv(paths['modified'] / 'train_modified.csv', index=False)
        
        logging.info("Modified datasets have been saved successfully.")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    setup_logging()
    main()