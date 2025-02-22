import pandas as pd
import os
import numpy as np

# Function to replace Turkish characters with English equivalents
def replace_turkish_chars(text):
    turkish_chars = {'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u'}
    for turkish, english in turkish_chars.items():
        text = text.replace(turkish, english)
    return text

# Function to preprocess station names
def preprocess_station_name(name):
    if pd.isna(name):
        return ""
    # Replace spaces with underscores
    name = name.strip().replace(' ', '_')
    # Replace NaN values
    name = name if name != "nan" else ""
    # Reduce multiple underscores to a single one
    while '__' in name:
        name = name.replace('__', '_')
    return name

# Load station coordinates and purposes
coord_file = "station_coordinates.xlsx"
coords_df = pd.read_excel(coord_file)

# Clean station names in the coordinates DataFrame
coords_df['station'] = coords_df['station'].apply(preprocess_station_name)
coords_df['station'] = coords_df['station'].apply(replace_turkish_chars)

# Directory containing the datasets
dataset_dir = "11th step ADDING NEW FEATURES (RH WS AND PM RATIO)/feature_added_dataset/"

# Directory to save the updated datasets
output_dir = "12th step ADDING COORDINATE OF STATION/coordinate_added_datasets/"
os.makedirs(output_dir, exist_ok=True)

# List all XLSX files in the directory
dataset_files = [f for f in os.listdir(dataset_dir) if f.endswith('.xlsx')]

# Extract and preprocess station names from the filenames
station_names = [os.path.splitext(f)[0].strip().lower().replace(' ', '_') for f in dataset_files]
station_names = [replace_turkish_chars(name) for name in station_names]

# Function to add longitude, latitude, and purpose columns
def add_long_lat_purpose(df, longitude, latitude):
    df['Longitude'] = longitude
    df['Latitude'] = latitude
    return df

# Process each dataset
for original_station, station in zip(dataset_files, station_names):
    dataset_path = os.path.join(dataset_dir, original_station)
    df = pd.read_excel(dataset_path)
    
    # Get the coordinates and purpose for the current station
    coords = coords_df[coords_df['station'] == station]
    if not coords.empty:
        longitude = coords.iloc[0]['longitude']
        latitude = coords.iloc[0]['latitude']
        
        # Add longitude, latitude, and purpose to the dataset
        df = add_long_lat_purpose(df, longitude, latitude)
        
        # Save the updated dataset
        output_path = os.path.join(output_dir, f'{original_station}')
        df.to_excel(output_path, index=False)
        
        print(f'Processed {station}')
    else:
        print(f'Coordinates not found for {station}')
