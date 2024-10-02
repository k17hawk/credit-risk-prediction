import pandas as pd
import sqlalchemy
import sqlite3
from sqlalchemy import create_engine, text


def create_mysql_database(username, password, host, database_name):
    """Creates a MySQL database if it doesn't already exist."""
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}/')
    with engine.connect() as connection:
        connection.execute(text(f"CREATE DATABASE IF NOT EXISTS {database_name}"))
    print(f"Database '{database_name}' created successfully.")


def connect_mysql(username, password, host, database_name):
    """Creates a connection to the MySQL database."""
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}/{database_name}')
    return engine


def load_csv_to_dataframe(csv_file_path):
    """Loads a CSV file into a pandas DataFrame."""
    df = pd.read_csv(csv_file_path)
    print("CSV file loaded into DataFrame.")
    return df


def save_to_mysql(df, engine, table_name):
    """Saves the DataFrame to a MySQL table."""
    df.to_sql(table_name, con=engine, if_exists='replace', index=False)
    print(f"Data inserted into MySQL table '{table_name}' successfully.")


def save_to_sqlite(df, sqlite_file, table_name):
    """Saves the DataFrame to an SQLite .db file."""
    conn = sqlite3.connect(sqlite_file)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Data inserted into SQLite database '{sqlite_file}' successfully.")


def export_csv_to_mysql_and_sqlite(csv_file_path, username, password, host, mysql_db_name, table_name, sqlite_db_file):
    """Main function to load CSV, save to MySQL and SQLite."""
    
    # Step 1: Create MySQL Database
    create_mysql_database(username, password, host, mysql_db_name)
    
    # Step 2: Load CSV into DataFrame
    df = load_csv_to_dataframe(csv_file_path)
    
    # Step 3: Save to MySQL
    mysql_engine = connect_mysql(username, password, host, mysql_db_name)
    save_to_mysql(df, mysql_engine, table_name)
    
    # Step 4: Save to SQLite
    save_to_sqlite(df, sqlite_db_file, table_name)


# Example usage
if __name__ == "__main__":
    # CSV file to be imported
    csv_file_path = "data.csv"
    
    # MySQL connection details
    username = "kumar"
    password = "kumar"
    host = "localhost"
    mysql_db_name = "credit_risk"
    
    # Table name
    table_name = "credit_risk"
    
    # SQLite file name
    sqlite_db_file = "credit_risk.db"
    
    # Call the main function to perform all tasks
    export_csv_to_mysql_and_sqlite(csv_file_path, username, password, host, mysql_db_name, table_name, sqlite_db_file)


