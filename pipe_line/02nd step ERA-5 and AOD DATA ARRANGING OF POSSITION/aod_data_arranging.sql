DO $$
DECLARE
    source_schema TEXT := 'aod_data';        -- Source schema name
    target_schema TEXT := 'aod_data_processed';    -- Target schema name
    rec RECORD;
    create_table_query TEXT;
BEGIN
    -- Loop through each table in the current source schema
    FOR rec IN
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = source_schema
    LOOP
        -- Generate CREATE TABLE query for each table in target schema
        create_table_query := 'CREATE TABLE IF NOT EXISTS ' || quote_ident(target_schema) || '.' || quote_ident(rec.table_name) || ' (
                              "Date" TIMESTAMP,
                              "Aerosol_Optical_Thickness_550_Land_Mean" NUMERIC,
                              "Aerosol_Optical_Thickness_550_Ocean_Mean" NUMERIC,
                              "Aerosol_Optical_Thickness_550_Ocean_Standard_Deviation" NUMERIC
                              )';
        
        -- Execute the CREATE TABLE query
        EXECUTE create_table_query;

        -- Construct the INSERT INTO query for the current table
        EXECUTE '
      INSERT INTO ' || quote_ident(target_schema) || '.' || quote_ident(rec.table_name) || ' ("Date", "Aerosol_Optical_Thickness_550_Land_Mean", "Aerosol_Optical_Thickness_550_Ocean_Mean", "Aerosol_Optical_Thickness_550_Ocean_Standard_Deviation")
      SELECT *
      FROM ' || quote_ident(source_schema) || '.crosstab(
          $query$SELECT "Date", "Variable", "Value"
          FROM ' || quote_ident(source_schema) || '.' || quote_ident(rec.table_name) || '
          ORDER BY 1, 2$query$,
          $query$VALUES (''Aerosol_Optical_Thickness_550_Land_Mean''), (''Aerosol_Optical_Thickness_550_Ocean_Mean''), (''Aerosol_Optical_Thickness_550_Ocean_Standard_Deviation'')$query$
      ) AS ct(
          "Date" TIMESTAMP,
          "Aerosol_Optical_Thickness_550_Land_Mean" NUMERIC,
          "Aerosol_Optical_Thickness_550_Ocean_Mean" NUMERIC,
          "Aerosol_Optical_Thickness_550_Ocean_Standard_Deviation" NUMERIC
      )';
    END LOOP;
END $$;
