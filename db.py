import os
import sys

import pandas as pd
import psycopg2


def connect_to_db()-> tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]:
    """
    Establishes a connection to a PostgreSQL database using environment variables for credentials.

    Environment variables used (with defaults if not set):
        - DBNAME: Database name (default: "stockspostgresdb")
        - USER: Username (default: "pguser")
        - PASSWORD: Password (default: "pgadmin")
        - HOST: Host address (default: "10.0.0.77")
        - PORT: Port number (default: "5432")

    Returns:
        tuple: A tuple containing the connection object and a database cursor.

    Raises:
        SystemExit: If the connection fails, logs the error to stderr and exits.
    """
    try:
        conn = psycopg2.connect(
            dbname=os.environ.get("DBNAME", "stockspostgresdb"),
            user=os.environ.get("USER", "pguser"),
            password=os.environ.get("PASSWORD", "pgadmin"),
            host=os.environ.get("HOST", "10.0.0.77"),
            port=os.environ.get("PORT", "5432")
        )
        return conn, conn.cursor()
    except psycopg2.Error as e:
        sys.stderr.write(f"Database connection error: {e}\n")
        sys.exit(1)

def fetch_data(cur: psycopg2.extensions.cursor, table_name: str)->pd.DataFrame:
    """
    Executes a SQL SELECT * query on the given table and returns the results as a pandas DataFrame.

    Args:
        cur (psycopg2.extensions.cursor): A PostgreSQL database cursor.
        table_name (str): Name of the table to fetch data from.

    Returns:
        pd.DataFrame: DataFrame containing all rows and columns from the specified table.

    Raises:
        Exception: Logs any exceptions to stderr during the query or DataFrame construction.
    """
    try:
        cur.execute(f"""
            SELECT * FROM {table_name}
        """)
        rows = cur.fetchall()
        col_names = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows, columns=col_names)
        return df
    except Exception as e:
        sys.stderr.write(f"Exception occured while fetching data: {e}\n")

def disconnect_from_db(conn: psycopg2.extensions.connection, cur: psycopg2.extensions.cursor)->None:
    """
    Safely closes the database cursor and connection.

    Parameters:
        conn (psycopg2.extensions.connection): The active database connection to close.
        cur (psycopg2.extensions.cursor): The active database cursor to close.

    Returns:
        None
    """
    if cur:
        cur.close()
    if conn:
        conn.close()