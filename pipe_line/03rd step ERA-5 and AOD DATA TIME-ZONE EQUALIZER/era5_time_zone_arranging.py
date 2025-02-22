import psycopg2
from psycopg2 import sql

# PostgreSQL database connection information
db_host = 'localhost'
db_name = 'postgres'
db_user = 'postgres'
db_password = 'postgres'
db_port = 5432

try:
    # Creating a connection
    with psycopg2.connect(
        host=db_host, 
        database=db_name, 
        user=db_user, 
        password=db_password, 
        port=db_port
    ) as conn:
        with conn.cursor() as cur:
            try:
                # Listing tables in the 'era5_data_processed' schema
                cur.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'era5_data_processed'
                """)
                
                tables = cur.fetchall()

                for table in tables:
                    table_name = table[0]
                    
                    # Checking if the table contains the 'Date' column
                    cur.execute(sql.SQL("""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = 'era5_data_processed'
                        AND table_name = %s
                        AND column_name = 'Date'
                    """), [table_name])
                    
                    if cur.fetchone() is not None:
                        # For each table, update the 'Date' column to Turkey timezone
                        update_query = sql.SQL("""
                            UPDATE {}.{}
                            SET "Date" = ("Date" AT TIME ZONE 'UTC' AT TIME ZONE 'Turkey')
                        """).format(
                            sql.Identifier('era5_data_processed'),
                            sql.Identifier(table_name)
                        )

                        # Running the update query
                        cur.execute(update_query)
                        conn.commit()

                        print(f"Updated table {table_name} successfully.")

                    else:
                        print(f"Table {table_name} does not contain 'Date' column.")

            except Exception as e:
                print("Error while processing tables:", e)

except Exception as e:
    print("Error connecting to the database:", e)
