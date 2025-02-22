import pandas as pd
import os
from datetime import datetime

def convert_date_format(date):
    # Convert Timestamp to datetime object
    dt = date.to_pydatetime()
    # Format datetime object as desired ('%m/%d/%Y')
    return dt.strftime('%m/%d/%Y')

def process_excel_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(input_folder, filename)
            print(f"Processing file: {file_path}")
            
            try:
                # Load Excel file
                df = pd.read_excel(file_path)
                
                # Assuming 'date' is the column containing the date/time to convert
                if 'date' in df.columns:
                    df['date'] = df['date'].apply(lambda x: convert_date_format(pd.Timestamp(x)))
                
                # Determine output file path
                new_filename = os.path.splitext(filename)[0] + '.xlsx'
                new_file_path = os.path.join(output_folder, new_filename)
                
                # Save modified dataframe back to Excel
                df.to_excel(new_file_path, index=False)
                
                print(f"File saved as: {new_file_path}")

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

# Main function
if __name__ == "__main__":
    input_folder = "07th step SIM DIVIDING DIFFERENT EXCEL FILES/dividing_different_excels/"
    output_folder = "08th step SIM REMOVING TIME FROM DATE COLUMN/removed_time_datasets/"
    process_excel_files(input_folder, output_folder)
