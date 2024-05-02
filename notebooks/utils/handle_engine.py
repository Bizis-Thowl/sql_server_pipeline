import os
from urllib.parse import quote

from dotenv import load_dotenv
from sqlalchemy import create_engine


def get_engine():
    server = 'localhost'
    database = 'metmast_0_4'

    # Construct the connection string with trusted connection
    connection_string = f'mssql+pyodbc://@{server}:1433/{database}?trusted_connection=yes&driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes' #TODO:Certificate
    engine = create_engine(connection_string)
    return engine


def get_engine2():
    load_dotenv()
    SERVER = os.getenv("SQL_SERVER_IP")
    PORT= os.getenv("SQL_SERVER_PORT")
    DATABASE = os.getenv("DB")
    USER = os.getenv("DB_USER")
    PASSWORD = os.getenv("DB_PW")
    DRIVER = os.getenv("DB_DRIVER")
    engine = create_engine(
        f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER}:{PORT}/{DATABASE}?driver={DRIVER}&TrustServerCertificate=yes")
    return engine
