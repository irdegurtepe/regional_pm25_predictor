import os
import pandas as pd
import numpy as np

# Define the input and output folder paths
input_folder_path = "10th step DATE SEPARATION/date_separated_datasets/"
output_folder_path = "11th step ADDING NEW FEATURES (RH WS AND PM RATIO)/feature_added_dataset/"

# Ensure the output directory exists
os.makedirs(output_folder_path, exist_ok=True)

# Formula for relative humidity calculation
def calculate_relative_humidity(temp, dew_point):
    return 100 * (np.exp((17.625 * dew_point) / (243.04 + dew_point)) / np.exp((17.625 * temp) / (243.04 + temp)))

# Process each Excel file in the input directory
for filename in os.listdir(input_folder_path):
    if filename.endswith(".xlsx"):
        input_file_path = os.path.join(input_folder_path, filename)
        
        # Load the Excel file into a DataFrame
        df = pd.read_excel(input_file_path)
        
        # Calculate the new features
        df["Wind Velocity"] = np.sqrt(df["10 metre U wind component"]**2 + df["10 metre V wind component"]**2)
        df['Relative Humidity'] = calculate_relative_humidity(df['2 metre temperature'], df['2 metre dewpoint temperature'])
        
        # Save the updated DataFrame to a new Excel file in the output directory
        output_file_path = os.path.join(output_folder_path, filename)
        df.to_excel(output_file_path, index=False)

        print(f"Processed and saved: {filename}")
