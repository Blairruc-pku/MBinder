CREATE OR REPLACE PROCEDURE get_buffer(table_name TEXT)
LANGUAGE plpython3u AS $$
    """
    Create a buffer for table by label partition
    """
    buffer_table_name = f"_{table_name}"
    query = "DROP TABLE IF EXISTS {}".format(buffer_table_name)
    plpy.execute(query)

    query = f"""
    CREATE TABLE {buffer_table_name} AS
    WITH stratified_sample AS (
    SELECT *
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY label ORDER BY random()) AS rn,
            COUNT(*) OVER (PARTITION BY label) AS total_count
        FROM {table_name}
    ) ranked
    WHERE rn <= CASE WHEN total_count*0.1 < 100 THEN total_count*0.1 ELSE 100 END
    )
    SELECT * FROM stratified_sample;
    """
    plpy.execute(query)
$$;