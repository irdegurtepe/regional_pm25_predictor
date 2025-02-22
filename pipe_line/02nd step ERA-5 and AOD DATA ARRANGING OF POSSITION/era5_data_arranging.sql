DO $$
DECLARE
    source_schema TEXT := 'era5_data';        -- Source schema name
    target_schema TEXT := 'era5_data_processed';    -- Target schema name
    rec RECORD;
    create_table_query TEXT;
    insert_into_query TEXT;
BEGIN
    -- Ensure the tablefunc module is available
    PERFORM 1 FROM pg_available_extensions WHERE name = 'tablefunc';
    IF NOT FOUND THEN
        RAISE EXCEPTION 'tablefunc module not found';
    END IF;

    -- Loop through each table in the current source schema
    FOR rec IN
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = source_schema
    LOOP
        -- Generate CREATE TABLE query for each table in target schema
        create_table_query := 
            'CREATE TABLE IF NOT EXISTS ' || quote_ident(target_schema) || '.' || quote_ident(rec.table_name) || ' 
            (
                "Date" TIMESTAMP,
                "10 metre U wind component" NUMERIC,
                "10 metre V wind component" NUMERIC,
                "2 metre temperature" NUMERIC,
                "Surface pressure" NUMERIC,
                "Total cloud cover" NUMERIC,
                "Total precipitation" NUMERIC,
                "2 metre dewpoint temperature" NUMERIC,
                "Boundary layer height" NUMERIC,
                "Forecast surface roughness" NUMERIC
            )';

        -- Execute the CREATE TABLE query
        EXECUTE create_table_query;

        -- Construct the INSERT INTO query for the current table
        insert_into_query := 
            'INSERT INTO ' || quote_ident(target_schema) || '.' || quote_ident(rec.table_name) || ' 
            (
                "Date",
                "10 metre U wind component", 
                "10 metre V wind component",
                "2 metre temperature", 
                "Surface pressure",
                "Total cloud cover",
                "Total precipitation", 
                "2 metre dewpoint temperature",
                "Boundary layer height",
                "Forecast surface roughness"
            )
            SELECT * FROM aod_data.crosstab
            (
                $query$SELECT "Date", "Variable", "Value"
                  FROM ' || quote_ident(source_schema) || '.' || quote_ident(rec.table_name) || '
                  ORDER BY 1, 2$query$,
                $query$VALUES 
                    (''10 metre U wind component''), 
                    (''10 metre V wind component''), 
                    (''2 metre temperature''), 
                    (''Surface pressure''), 
                    (''Total cloud cover''), 
                    (''Total precipitation''), 
                    (''2 metre dewpoint temperature''), 
                    (''Boundary layer height''), 
                    (''Forecast surface roughness'')$query$
            ) 
            AS ct(
                "Date" TIMESTAMP,
                "10 metre U wind component" NUMERIC,
                "10 metre V wind component" NUMERIC,
                "2 metre temperature" NUMERIC,
                "Surface pressure" NUMERIC,
                "Total cloud cover" NUMERIC,
                "Total precipitation" NUMERIC,
                "2 metre dewpoint temperature" NUMERIC,
                "Boundary layer height" NUMERIC,
                "Forecast surface roughness" NUMERIC
            )';

        -- Execute the INSERT INTO query
        EXECUTE insert_into_query;
    END LOOP;
END $$;
