import os
import re
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import KDTree
from unidecode import unidecode
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import logging
import multiprocessing as mp

# Define a function to set up logging configuration
def setup_logging():
    """Sets up logging configuration to only show info level messages."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a function to get NetCDF file paths from a folder
def get_nc_file_paths(folder_path):
    """Retrieves NetCDF file paths from the specified folder."""
    nc_file_paths = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files if file.endswith(".nc")]
    logging.info(f'Found {len(nc_file_paths)} NetCDF files')
    return nc_file_paths

# Define a function to read and process station data from an Excel file
def read_and_process_station_data(excel_file):
    """Reads station data from an Excel file and processes the station names."""
    logging.debug(f'Reading Excel file from {excel_file}')
    stations_df = pd.read_excel(excel_file)
    stations_df['station'] = stations_df['station'].apply(lambda x: re.sub(r'\s+', '_', unidecode(x.lower())))
    logging.info(f'Processed {len(stations_df)} station names')
    return stations_df

# Define a function to convert the station DataFrame to a dictionary
def stations_df_to_dict(stations_df):
    """Converts the station DataFrame to a dictionary with station names and coordinates."""
    stations = stations_df.set_index('station').T.to_dict('list')
    logging.debug('Converted stations DataFrame to dictionary')
    return stations

# Define a function to extract the date from the NetCDF filename
def extract_date_from_filename(filename):
    """Extracts the date from the NetCDF filename."""
    match = re.search(r'A(\d{4})(\d{3})', filename) 
    if match:
        year = int(match.group(1))
        day_of_year = int(match.group(2))
        date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
        return date
    else:
        return None

def process_nc_files_for_station(nc_file_paths, target_lat, target_lon):
    """Processes the NetCDF files for a specific station and extracts relevant data."""
    df_station = pd.DataFrame()

    # Define the variables of interest
    variables_of_interest = [
        'Aerosol_Optical_Thickness_550_Land_Mean',
        'Aerosol_Optical_Thickness_550_Ocean_Mean',
        'Aerosol_Optical_Thickness_550_Ocean_Standard_Deviation'
    ]

    for nc_file in nc_file_paths:
        logging.debug(f'Opening NetCDF file: {nc_file}')
        ds = xr.open_dataset(nc_file)

        # Extract the date from the filename
        date = extract_date_from_filename(nc_file)
        if date is None:
            logging.error(f"Could not extract date from filename: {nc_file}")
            continue

        # Checking for the presence of latitude and longitude variables
        possible_lat_keys = ['lat_nc', 'latitude', 'Latitude', 'lat', 'Lat']
        possible_lon_keys = ['lon_nc', 'longitude', 'Longitude', 'lon', 'Lon']

        # Find the latitude and longitude keys in the dataset
        lat_key = next((key for key in possible_lat_keys if key in ds.variables), None)
        lon_key = next((key for key in possible_lon_keys if key in ds.variables), None)

        # If latitude and longitude keys are not found, log an error and skip the file
        if not lat_key or not lon_key:
            logging.error(f"Neither latitude nor longitude key found in the file: {nc_file}")
            continue
        
        # Extract latitude and longitude values
        lats = ds[lat_key].values
        lons = ds[lon_key].values

        # Flatten the latitude and longitude arrays
        lats_flat = lats.ravel()
        lons_flat = lons.ravel()

        # Create a KDTree for finding the nearest grid points
        tree = KDTree(list(zip(lats_flat, lons_flat)))
        _, indices = tree.query([(target_lat, target_lon)], k=4)
        indices = indices[0]

        # Initialize a list to store the data
        data_list = []

        # Extract data for each variable of interest
        for var_name in variables_of_interest:
            if var_name not in ds.variables:
                logging.warning(f"Variable {var_name} not found in the file: {nc_file}")
                continue

            data = ds[var_name].values
            data_flat = data.ravel()

            # Handle the case where indices may be out of bounds
            if len(indices) > 0 and indices[0] < len(data_flat):
                weights = 1 / np.sqrt((lats_flat[indices] - target_lat) ** 2 + (lons_flat[indices] - target_lon) ** 2)
                weighted_avg = np.average(data_flat[indices], weights=weights)
            else:
                weighted_avg = np.nan  # Handle case where there are no valid indices

            data_list.append([date, var_name, weighted_avg])

        df_temp = pd.DataFrame(data_list, columns=['Date', 'Variable', 'Value'])
        df_station = pd.concat([df_station, df_temp], ignore_index=True)

    return df_station

# Define a function to combine the results from processing chunks of stations
def combine_results(results):
    """Combines the results from processing chunks of stations."""
    dfs_per_station = {}
    for result in results:
        for station, df_station in result.items():
            if station not in dfs_per_station:
                dfs_per_station[station] = df_station
            else:
                dfs_per_station[station] = pd.concat([dfs_per_station[station], df_station], ignore_index=True)
    return dfs_per_station

# Define a worker function to process a chunk of stations
def worker(nc_file_paths, station_chunk):
    """Worker function to process a chunk of stations."""
    chunk_results = {station: pd.DataFrame() for station in station_chunk.keys()}
    
    # Process each station in the chunk
    for station, (target_lat, target_lon) in station_chunk.items():
        logging.info(f'Processing station: {station}')
        df_station = process_nc_files_for_station(nc_file_paths, target_lat, target_lon)
        if not chunk_results[station].empty:
            chunk_results[station] = pd.concat([chunk_results[station], df_station], ignore_index=True)
        else:
            chunk_results[station] = df_station

    logging.info(f'Finished processing chunk of stations: {list(station_chunk.keys())}')
    return chunk_results

# Define a function to process NetCDF files in parallel by stations
def process_nc_files_in_parallel(nc_file_paths, stations):
    """Splits and processes NetCDF files in parallel by stations."""
    num_chunks = min(mp.cpu_count(), len(stations))
    chunks = [dict(list(stations.items())[i::num_chunks]) for i in range(num_chunks)]
    
    # Process chunks in parallel
    logging.info(f'Split stations into {num_chunks} chunks for processing.')
    with mp.Pool(processes=num_chunks) as pool:
        results = pool.starmap(worker, [(nc_file_paths, chunk) for chunk in chunks])

    # Combine results
    dfs_per_station = combine_results(results)
    return dfs_per_station

# Define a function to establish a connection to the PostgreSQL database
def connect_to_database(db_params):
    """Establishes a connection to the PostgreSQL database."""
    try:
        engine = create_engine(f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}')
        conn = engine.raw_connection()
        cur = conn.cursor()
        return conn, cur, engine
    except Exception as e:
        logging.error(f'An error occurred while connecting to the database: {e}')
        raise

# Define a function to save the DataFrames to the database
def save_to_database(dfs_per_station, engine):
    """Saves the DataFrames to the database."""
    try:
        for station, df in dfs_per_station.items():
            df.to_sql(station, engine, schema="aod_data", if_exists='replace', index=False)
        logging.info("Data has been successfully imported to the database.")
    except Exception as e:
        logging.error(f'An error occurred while saving data: {e}')

def main():
    setup_logging()
    
    # Folder path for the NetCDF files
    folder_path = "example_aot_data"
    
    # Excel file path
    excel_file = "station_coordinates.xlsx"

    # Get NetCDF file paths
    nc_file_paths = get_nc_file_paths(folder_path)

    # Read and process station data
    stations_df = read_and_process_station_data(excel_file)

    # Convert station DataFrame to dictionary
    stations = stations_df_to_dict(stations_df)

    # Process all stations
    dfs_per_station = process_nc_files_in_parallel(nc_file_paths, stations)

    # Database connection parameters (default values)
    db_params = {
        "host": "localhost",
        "database": "postgres",
        "user": "postgres",
        "password": "postgres",
        "port": "5432",
        "schema": "aod_data"
    }

    try:
        # Connect to database
        conn, cur, engine = connect_to_database(db_params)

        # Insert data into database
        save_to_database(dfs_per_station, engine)

        # Close the connection
        cur.close()
        conn.close()

    except Exception as e:
        logging.error(f'An error occurred during the process: {e}')

# Run the main function
if __name__ == "__main__":
    main()