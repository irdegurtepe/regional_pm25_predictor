import os
import pandas as pd

# Define the input and output directories
input_folder = "04th step ERA-5 and AOD DATA EXPORT DATA/era5_datasets/"
output_folder = "05th step ERA-5 and AOD DATA GET DAILY AVG/era5_datasets/"

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Process each XLSX file in the input directory
for filename in os.listdir(input_folder):
    if filename.endswith(".xlsx"):
        # Load the XLSX file
        file_path = os.path.join(input_folder, filename)
        df = pd.read_excel(file_path)
        
        # Rename the 'Date' column to 'date'
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'date'}, inplace=True)
        
        # Drop rows with invalid dates
        df.dropna(subset=['date'], inplace=True)
        
        # Convert the 'date' column to datetime, allowing for mixed formats
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Drop rows with invalid dates after conversion
        df.dropna(subset=['date'], inplace=True)
        
        # Filter out dates outside the range 2018-2023
        df = df[(df['date'] >= '2018-01-01') & (df['date'] <= '2023-12-31')]
        
        # Sort the DataFrame by 'date' in ascending order
        df.sort_values(by='date', inplace=True)
        
        # Set the 'date' column as the index
        df.set_index('date', inplace=True)
        
        # Resample the data to daily frequency, taking the mean of each day
        daily_df = df.resample('D').mean()
        
        # Reset the index to make 'date' a column again
        daily_df.reset_index(inplace=True)
        
        # Save the daily averages to the output folder as an XLSX file
        output_file_path = os.path.join(output_folder, filename)
        daily_df.to_excel(output_file_path, index=False)

print("Processing complete. Daily averages have been saved to the output folder.")
