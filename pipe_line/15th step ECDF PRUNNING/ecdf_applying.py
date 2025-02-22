import pandas as pd
import numpy as np
import os

# Define the input and output folder paths
input_folder = "14th step ORDERED DATASET/ordered_dataset/"
output_folder = "15th step ECDF PRUNNING/pruned_dataset/"

# Define the columns to be processed (pollutant columns)
columns_to_process = ['PM10', 'PM2.5']

# Define a function to calculate ECDF
def ecdf(data):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    y = np.arange(1, n + 1) / n
    return sorted_data, y

# Define a function to drop values close to 1 or 0 based on ECDF results
def drop_ecdf_extremes(df, threshold=0.001):
    # Ensure only the specified columns are processed
    for col in columns_to_process:
        if col in df.columns:
            # Calculate ECDF for the column
            sorted_data, y_ecdf = ecdf(df[col].dropna())

            # Identify values close to 1 or 0
            close_to_zero = sorted_data[y_ecdf <= threshold]
            close_to_one = sorted_data[y_ecdf >= (1 - threshold)]

            # Replace cells with NaN where the values are close to 1 or 0
            df[col].replace(close_to_zero, np.nan, inplace=True)
            df[col].replace(close_to_one, np.nan, inplace=True)
        
    return df

# Process all Excel files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.xlsx'):
        input_file = os.path.join(input_folder, file_name)
        output_file = os.path.join(output_folder, file_name)
        
        # Read the Excel file
        df = pd.read_excel(input_file)
        
        # Apply the function to drop values close to 1 or 0 in ECDF for specific columns
        df_cleaned = drop_ecdf_extremes(df, threshold=0.001)
        
        # Save the cleaned dataset
        df_cleaned.to_excel(output_file, index=False)
        
        print(f"Cleaned dataset saved to {output_file}")
