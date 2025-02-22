import pandas as pd
import os
import numpy as np

# Directory containing the datasets with separated date columns
input_dir = "12th step ADDING COORDINATE OF STATION/coordinate_added_datasets/"

# Directory to save the merged dataset
output_dir = "13th step MERGING ALL STATION IN ONE EXCEL/merged_dataset/"
os.makedirs(output_dir, exist_ok=True)

# List all Excel files in the input directory
dataset_files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx')]

if not dataset_files:
    print("No Excel files found in the input directory.")
else:
    # Initialize an empty list to store DataFrames
    dataframes = []

    # Read and filter each dataset
    for dataset_file in dataset_files:
        dataset_path = os.path.join(input_dir, dataset_file)
        try:
            df = pd.read_excel(dataset_path)

            # Check if the dataset has 'PM10' and 'PM2.5' columns
            if 'PM10' in df.columns and 'PM2.5' not in df.columns:
                
                
                # Drop Some columns if they exist
                columns_to_drop = ['O3', 'SO2','NOX', 'NO', 'NO2', 'CO']
                df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
                
                """               
                # Drop rows where PM2.5 is greater than PM10
                df = df[df['PM10'] >= df['PM2.5']]
                
                # Drop rows where PM10 is greater than 10 times PM2.5
                df = df[df['PM10'] <= 10 * df['PM2.5']]
                """
                
                """
                # Drop rows where the sum of NO2 and NO is greater than NOX              
                if 'NO2' in df.columns and 'NO' in df.columns and 'NOX' in df.columns:
                    df = df[(df['NO2'] + df['NO']) <= df['NOX']]
                    # Drop rows where the sum of NO2 and NO do not match with NOX properly
                    df = df[df['NOX'] <= 2 * (df['NO2'] + df['NO'])]
                """
                """
                # Replace cells where PM2.5 is greater than PM10 with NaN
                df.loc[df['PM10'] < df['PM2.5'], ['PM10', 'PM2.5']] = np.nan
                
                # Replace cells where PM10 is greater than 10 times PM2.5 with NaN
                df.loc[df['PM10'] > 10 * df['PM2.5'], ['PM10', 'PM2.5']] = np.nan
                """
                """
                # Replace cells where the sum of NO2 and NO is greater than NOX with NaN
                if 'NO2' in df.columns and 'NO' in df.columns and 'NOX' in df.columns:
                    df.loc[(df['NO2'] + df['NO']) > df['NOX'], ['NO2', 'NO', 'NOX']] = np.nan
                    
                    # Replace cells where NOX is not within the proper range of the sum of NO2 and NO with NaN
                    df.loc[df['NOX'] > 2 * (df['NO2'] + df['NO']), ['NO2', 'NO', 'NOX']] = np.nan
                """
                
                               
                # Drop rows with more than 4 empty cells
                # df = df.dropna(thresh=len(df.columns) - 3)
                
                
                # Replace the invalid Total Precipitation value (-0.0000000009313226) with NaN
                if 'Total precipitation' in df.columns:
                    df['Total precipitation'].replace(-0.0000000009313226, np.nan, inplace=True)
                
                # Remove rows where there are 15 consecutive NaN values in either 'PM2.5' or 'PM10'
                def drop_consecutive_nan_blocks(df, column, block_size):
                    # Create a rolling window over the column and count NaN values in each block
                    mask = df[column].isna().rolling(window=block_size, min_periods=block_size).sum() == block_size
                    # Drop rows that belong to blocks with consecutive NaNs
                    return df.loc[~mask]

                #df = drop_consecutive_nan_blocks(df, 'PM2.5', 10)
                df = drop_consecutive_nan_blocks(df, 'PM10', 10)
                
                """
                # PMs ratio adding     
                if 'PM10' in df.columns and 'PM2.5' in df.columns:
                    df['PM10/PM2.5'] = df['PM10'] / df['PM2.5']
                """
                
                # Append the filtered DataFrame to the list
                dataframes.append(df)

        except Exception as e:
            print(f"Error reading {dataset_file}: {e}")

    # Concatenate all DataFrames into a single DataFrame
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)

        # Save the merged DataFrame to an Excel file
        output_path = os.path.join(output_dir, 'merged_dataset.xlsx')
        merged_df.to_excel(output_path, index=False)

        print(f'Merged dataset saved to {output_path}')
    else:
        print("No dataframes to merge.")
