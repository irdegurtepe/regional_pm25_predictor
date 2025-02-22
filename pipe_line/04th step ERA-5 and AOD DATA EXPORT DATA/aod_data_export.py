import psycopg2
import os
import logging
from psycopg2 import sql
from openpyxl import Workbook

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# PostgreSQL connection parameters
dbname = 'postgres'
user = 'postgres'
password = 'postgres'
host = 'localhost'
port = '5432'
schema_name = 'aod_data_processed'

# Output folder path for XLSX files
output_folder = "04th step ERA-5 and AOD DATA EXPORT DATA/aod_datasets/"

def export_table_to_xlsx(conn, schema_name, table_name, output_folder, batch_size=1000):
    cursor = conn.cursor()

    try:
        # Query to select all rows from the table
        query = sql.SQL("SELECT * FROM {}.{}").format(
            sql.Identifier(schema_name),
            sql.Identifier(table_name)
        )
        cursor.execute(query)
        
        # Fetch the column names
        col_names = [desc[0] for desc in cursor.description]
        
        # Ensure output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Define XLSX file path
        xlsx_file_path = os.path.join(output_folder, f"{table_name}.xlsx")
        
        # Create a workbook and select the active worksheet
        wb = Workbook()
        ws = wb.active
        ws.title = table_name
        
        # Write header
        ws.append(col_names)
        
        while True:
            # Fetch rows in batches
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            # Write rows
            for row in rows:
                ws.append(row)
        
        # Save the workbook
        wb.save(xlsx_file_path)
        
        logging.info(f"Table '{table_name}' exported to '{os.path.abspath(xlsx_file_path)}'")
    except Exception as e:
        logging.error(f"Failed to export table '{table_name}': {e}")
    finally:
        cursor.close()

try:
    # Establish a connection to the database
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
    
    # Create a cursor object using the cursor() method
    cursor = conn.cursor()

    # Get all table names in the schema
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = %s
    """, (schema_name,))
    
    table_names = cursor.fetchall()

    # Iterate through each table and export data
    for table in table_names:
        table_name = table[0]
        try:
            export_table_to_xlsx(conn, schema_name, table_name, output_folder)
        except Exception as e:
            logging.error(f"Error processing table '{table_name}': {e}")

finally:
    # Close cursor and connection
    cursor.close()
    conn.close()
