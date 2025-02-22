import pandas as pd
import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up logging configuration
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths to your folders containing Excel files
aod_data = "05th step ERA-5 and AOD DATA GET DAILY AVG/aod_datasets/"
era5_data = "05th step ERA-5 and AOD DATA GET DAILY AVG/era5_datasets/"
sim_data = "08th step SIM REMOVING TIME FROM DATE COLUMN/removed_time_datasets/"
pandemic = "date_of_pandemic_daily.xlsx"

# Save each merged DataFrame with its respective file name to the specified output folder
output_folder = "09th step MERGING ERA-5 AOD AND SIM DATA/merged_datasets/"

# Function to read all Excel files from a folder
def read_excel_files(folder_path):
    excel_files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx')]
    dfs = []
    for file in excel_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_excel(file_path)
        df['date'] = pd.to_datetime(df['date'])  # Convert 'date' column to datetime
        dfs.append((file.split('.')[0], df))  # Store file name without extension along with DataFrame
    return dfs

# Read Excel files from the respective folders
dfs_aod = read_excel_files(aod_data)
dfs_era5 = read_excel_files(era5_data)
dfs_sim = read_excel_files(sim_data)

# Function to merge DataFrames based on file names
def merge_dataframes(sim_name, sim_df):
    logging.info(f"Processing station: {sim_name}")
    
    # Find matching AOD and ERA5 files
    aod_df = next((df for name, df in dfs_aod if name == sim_name), None)
    era5_df = next((df for name, df in dfs_era5 if name == sim_name), None)
    
    if aod_df is not None and era5_df is not None:
        # Merge sim_df with era5_df and aod_df, ensuring sim_df's shape is preserved
        merged_df = pd.merge(sim_df, era5_df, on='date', how='left')
        merged_df = pd.merge(merged_df, aod_df, on='date', how='left')
        merged_df = pd.merge(merged_df, pd.read_excel(pandemic), on='date', how='left')
        return sim_name, merged_df
    return None

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process the merging in parallel
with ProcessPoolExecutor() as executor:
    future_to_name = {executor.submit(merge_dataframes, sim_name, sim_df): sim_name for sim_name, sim_df in dfs_sim}
    
    for future in as_completed(future_to_name):
        result = future.result()
        if result:
            file_name, df = result
            output_file_path = os.path.join(output_folder, f"{file_name}.xlsx")
            df.to_excel(output_file_path, index=False)
            logging.info(f"Successfully processed and saved {file_name}.xlsx")

# Final log to indicate completion of all tasks
logging.info("All files have been processed and saved.")
