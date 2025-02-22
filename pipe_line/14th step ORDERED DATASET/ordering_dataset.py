import pandas as pd
import os
from datetime import datetime

# Directories
input_dir = "13th step MERGING ALL STATION IN ONE EXCEL/merged_dataset/"
output_dir = "14th step ORDERED DATASET/ordered_dataset/"
os.makedirs(output_dir, exist_ok=True)

# Function to convert date format
def convert_date_format(date):
    # Convert Timestamp to datetime object
    dt = date.to_pydatetime()
    # Format datetime object as desired ('%m/%d/%Y')
    return dt.strftime('%m/%d/%Y')

def process_file(file_path, output_path):
    # Load the dataset
    df = pd.read_excel(file_path)
    
    # Replace "_" with " " 
    df.columns = [col.replace('_', ' ') for col in df.columns]
    
    # Define column groups
    date_columns = ['date', "Year", "Month", "Day"]
    # pollutant_columns = ['PM10', 'PM2.5', 'NO', 'NO2', 'NOX', 'CO', 'SO2', 'O3']
    pollutant_columns = ['PM10']
    
    # Check if the primary date column exists in the DataFrame
    if 'date' in df.columns:
        other_columns = [col for col in df.columns if col not in pollutant_columns + date_columns]
        ordered_columns = date_columns + pollutant_columns + other_columns
        
        # Reorder columns, ensuring all columns exist in the DataFrame
        ordered_columns = [col for col in ordered_columns if col in df.columns]
        df = df[ordered_columns]
        
        # Convert date format and sort by date column
        df['date'] = df['date'].apply(convert_date_format)
        df = df.sort_values(by='date')

        # Save the reordered and sorted dataset
        df.to_excel(output_path, index=False)
    else:
        print(f"Date column missing in file: {file_path}")

# Process each file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".xlsx"):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        process_file(input_file, output_file)

print(f"All files have been processed and saved to {output_dir}")
