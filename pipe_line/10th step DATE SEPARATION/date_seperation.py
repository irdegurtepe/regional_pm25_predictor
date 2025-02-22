import pandas as pd
import os

# Function to replace Turkish characters with English equivalents
def replace_turkish_chars(text):
    turkish_chars = {'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u'}
    for turkish, english in turkish_chars.items():
        text = text.replace(turkish, english)
    return text

# Directory containing the pre-processed datasets with added coordinates
input_dir = "09th step MERGING ERA-5 AOD AND SIM DATA/merged_datasets/"

# Directory to save the updated datasets with separated date columns
output_dir = "10th step DATE SEPARATION/date_separated_datasets/"
os.makedirs(output_dir, exist_ok=True)

# List all XLSX files in the directory
dataset_files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx')]

# Function to add year, month, and day columns and remove the date column
def separate_date(df):
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Handle invalid dates gracefully
        df['Year'] = df['date'].dt.year
        df['Month'] = df['date'].dt.month
        df['Day'] = df['date'].dt.day
        return df
    else:
        print("No 'date' column found in the dataset")
        return df

# Process each dataset
for dataset_file in dataset_files:
    dataset_path = os.path.join(input_dir, dataset_file)
    df = pd.read_excel(dataset_path)
    
    # Separate the date column
    df = separate_date(df)
    
    # Save the updated dataset
    output_path = os.path.join(output_dir, dataset_file)
    df.to_excel(output_path, index=False)
    
    print(f'Processed {dataset_file}')
