import os
import re
import numpy as np
import pandas as pd
import xarray as xr
from unidecode import unidecode
from sqlalchemy import create_engine
import logging

# Define a function to set up logging configuration
def setup_logging():
    """Sets up logging configuration."""
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a function to get NetCDF file paths from a folder
def get_nc_file_paths(folder_path):
    """Retrieves NetCDF file paths from the specified folder."""
    nc_file_paths = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files if file.endswith(".nc")]
    logging.info(f'Found {len(nc_file_paths)} NetCDF files')
    return nc_file_paths

# Define a function to read and process the station data from an Excel file
def read_station_data(excel_file):
    """Reads and processes the station data from an Excel file."""
    logging.debug(f'Reading Excel file from {excel_file}')
    stations_df = pd.read_excel(excel_file)
    stations_df['station'] = stations_df['station'].apply(lambda x: re.sub(r'\s+', '_', unidecode(x.lower())))
    logging.info(f'Processed {len(stations_df)} station names')
    return stations_df.set_index('station').T.to_dict('list')

# Define a function to process a NetCDF file
def process_nc_file(nc_file, stations):
    """Processes a NetCDF file and returns results for each station."""
    results = {station: [] for station in stations.keys()}
    try:
        logging.info(f'Started processing NetCDF file: {nc_file}')
        ds = xr.open_dataset(nc_file)
        
        # Extract the latitude and longitude values
        lats = ds.latitude.values
        lons = ds.longitude.values
        
        # Create a meshgrid of latitude and longitude values
        lats, lons = np.meshgrid(lats, lons, indexing='ij')
        
        # Iterate over each variable in the NetCDF file
        for variable_name in ds.data_vars.keys():
            variable_data = ds[variable_name].values
            parameter_name = ds[variable_name].attrs.get('long_name', variable_name)

            # Iterate over each station and calculate the weighted average
            for station, (target_lat, target_lon) in stations.items():
                distances = np.sqrt((lats - target_lat)**2 + (lons - target_lon)**2)
                nearest_indices = np.unravel_index(np.argpartition(distances.ravel(), 4)[:4], distances.shape)
                nearest_distances = distances[nearest_indices]
                nearest_values = variable_data[:, nearest_indices[0], nearest_indices[1]]
                
                # Calculate the weighted average
                weights = 1 / nearest_distances
                weights /= weights.sum()
                weighted_avg = np.average(nearest_values, axis=1, weights=weights)
                times = pd.to_datetime(ds.time.values)
                
                # Append the results to the dictionary
                for i, dataDateTime in enumerate(times):
                    results[station].append({
                        'Date': dataDateTime,
                        'Variable': parameter_name,
                        'Value': weighted_avg[i]
                    })
        logging.info(f'Finished processing NetCDF file: {nc_file}')
    except Exception as e:
        logging.error(f'Error processing NetCDF file {nc_file}: {e}')
    return results

# Define a function to save the processed results to the database
def save_to_database(results, db_params):
    """Saves the processed results directly to the database."""
    try:
        engine = create_engine(f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}')
        for station, df_station_list in results.items():
            df_station = pd.DataFrame(df_station_list)
            df_station.to_sql(station, engine, schema="era5_data", if_exists='append', index=False)
        logging.info("Data has been successfully imported to the database.")
    except Exception as e:
        logging.error(f'An error occurred: {e}')

# Define a function to process a NetCDF file and save the results to the database
def process_nc_file_and_save(nc_file, stations, db_params):
    """Processes a NetCDF file and saves the results directly to the database."""
    results = process_nc_file(nc_file, stations)
    save_to_database(results, db_params)

# Define a function to process NetCDF files individually and save the results to the database
def process_nc_files_individually(nc_file_paths, stations, db_params):
    """Processes each NetCDF file individually and saves the results to the database."""
    for nc_file in nc_file_paths:
        process_nc_file_and_save(nc_file, stations, db_params)

# Define the main function
def main():
    setup_logging()
    
    folder_path = "example_met_data"
    excel_file = "station_coordinates.xlsx"
    
    nc_file_paths = get_nc_file_paths(folder_path)
    logging.info(f'Found {len(nc_file_paths)} NetCDF files')
    
    stations = read_station_data(excel_file)
    
    # Define the database parameters (default values)
    db_params = {
      "host": "localhost",
      "database": "postgres",
      "user": "postgres",
      "password": "postgres",
      "port": "5432",
      "schema": "era5_data"
    }

    # Process the NetCDF files individually and save the results to the database
    process_nc_files_individually(nc_file_paths, stations, db_params)

# Call the main function
if __name__ == "__main__":
    main()