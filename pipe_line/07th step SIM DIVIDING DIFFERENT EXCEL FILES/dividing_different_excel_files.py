import os
import pandas as pd

# Paths
input_file_path = "06th step SIM DIVDING DIFFERENT EXCEL SHEETS/divided_differnt_excel_sheets/example_monitoring_results.xlsx"
output_folder_path = "07th step SIM DIVIDING DIFFERENT EXCEL FILES/dividing_different_excels/"

# Create output directory if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

try:
    # Open the Excel file
    excel_file = pd.ExcelFile(input_file_path, engine="openpyxl")
    print(f"Opened Excel file: {input_file_path}")
    
    # Define substrings to remove from column names
    substrings_to_remove = [" ( µg/m3 )", " ( ppb )", " ( ug/m3 )", " ( mg/m3 )", " ( - )", " ( l/dk )", " ( ppm )", " "]
    
    # Iterate through each sheet one by one
    for sheet_name in excel_file.sheet_names:
        print(f"Processing sheet: {sheet_name}")
        
        # Read the sheet into a DataFrame
        df = pd.read_excel(input_file_path, sheet_name=sheet_name, engine="openpyxl")
        print(f"DataFrame loaded for sheet: {sheet_name}")
        
        # Modify the column names by removing specific substrings
        for substring in substrings_to_remove:
            df.columns = df.columns.str.replace(substring, "", regex=False)
        
        # Debugging: Print column names after cleaning
        print(f"Cleaned column names: {df.columns.tolist()}")
        
        # Convert periods to commas in the DataFrame values
        for col in df.columns:
            if df[col].dtype == 'object':  # Check if column contains string values
                df[col] = df[col].apply(lambda x: str(x).replace('.', '') if isinstance(x, str) else x)
        
        # Prepare the Excel file name
        excel_file_path = os.path.join(output_folder_path, f"{sheet_name}.xlsx")
        
        # Save the DataFrame as an Excel file
        df.to_excel(excel_file_path, index=False)
        print(f"Saved Excel file: {excel_file_path}")
    
    print("All sheets processed and saved as Excel files.")
except Exception as e:
    print(f"An error occurred: {e}")
