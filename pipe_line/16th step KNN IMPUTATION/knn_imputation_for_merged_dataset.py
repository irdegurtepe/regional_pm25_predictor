import os
import pandas as pd
from sklearn.impute import KNNImputer
import multiprocessing as mp
import logging
import numpy as np
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO)

# Function to perform KNN imputation on a chunk of the DataFrame
def knn_impute_chunk(chunk, numeric_cols, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_chunk = pd.DataFrame(imputer.fit_transform(chunk[numeric_cols]), columns=numeric_cols)
    return imputed_chunk

# Function to identify dummy variables
def identify_dummy_variables(df):
    dummy_cols = ["Year", "Month", "Day", "COVID - 19"]
    return dummy_cols

# Function to process a single chunk
def process_chunk(chunk_info):
    chunk, numeric_cols, n_neighbors = chunk_info
    return knn_impute_chunk(chunk, numeric_cols, n_neighbors)

# Function to dynamically manage CPU and RAM usage
def process_chunks_with_dynamic_memory(df, numeric_cols, n_neighbors=5, max_ram_usage=0.8):
    # Get the number of available CPUs
    num_cpus = mp.cpu_count()

    # Estimate available memory and adjust chunk size
    available_memory = psutil.virtual_memory().available
    chunk_size = int(len(df) / num_cpus)
    
    # Adjust chunk size based on RAM availability
    while True:
        estimated_memory_usage = chunk_size * df[numeric_cols].memory_usage(deep=True).sum()
        if estimated_memory_usage <= available_memory * max_ram_usage:
            break
        chunk_size = max(1, chunk_size // 2)

    # Split DataFrame into smaller chunks
    chunks = np.array_split(df, max(1, len(df) // chunk_size))

    # Create a pool of workers
    with mp.Pool(processes=num_cpus) as pool:
        # Process each chunk in parallel
        imputed_chunks = pool.map(process_chunk, [(chunk, numeric_cols, n_neighbors) for chunk in chunks])

    # Combine imputed chunks back into a single DataFrame
    df_imputed = pd.concat(imputed_chunks, axis=0).reset_index(drop=True)
    return df_imputed

# Function to process the file
def process_file(file_path, output_folder, max_ram_usage=0.8):
    try:
        logging.info(f"Starting processing for {os.path.basename(file_path)}")

        # Load dataset
        df = pd.read_excel(file_path)
        
        # Strip any leading/trailing spaces from column names
        df.columns = df.columns.str.strip()

        # Identify and drop dummy variables
        dummy_cols = identify_dummy_variables(df)
        df_cleaned = df.drop(columns=dummy_cols)

        # Separate numeric and non-numeric columns
        numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
        non_numeric_cols = df_cleaned.select_dtypes(exclude=['number']).columns

        # Perform KNN imputation on numeric columns with dynamic memory management
        df_numeric_imputed = process_chunks_with_dynamic_memory(df_cleaned, numeric_cols, max_ram_usage=max_ram_usage)

        # Combine imputed numeric columns with non-numeric columns
        df_imputed = pd.concat([df[non_numeric_cols].reset_index(drop=True), df_numeric_imputed], axis=1)

        # Add the dummy columns back to the DataFrame
        df_imputed = pd.concat([df_imputed, df[dummy_cols].reset_index(drop=True)], axis=1)

        # Save the imputed DataFrame to a new Excel file in the output folder
        output_filename = os.path.join(output_folder, os.path.basename(file_path).replace(".xlsx", "_imputed.xlsx"))
        df_imputed.to_excel(output_filename, index=False)

        logging.info(f"Completed processing for {os.path.basename(file_path)}")
    except Exception as e:
        logging.error(f"Error processing {os.path.basename(file_path)}: {e}")

# Main function to run the processing
def main():
    # Input and output folders
    input_folder = "15th step ECDF PRUNNING/pruned_dataset/"
    output_folder = "16th step KNN IMPUTATION/imputed_dataset/"
    
    # Ensure output folder exists, create if not
    os.makedirs(output_folder, exist_ok=True)

    # Get the single Excel file in the input folder
    file_path = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.endswith(".xlsx")][0]

    # Process the file
    process_file(file_path, output_folder)

    logging.info("File processed and imputed.")

if __name__ == "__main__":
    main()
