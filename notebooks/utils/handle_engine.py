from sqlalchemy import create_engine

from dotenv import load_dotenv
import os
load_dotenv()

def get_engine():
    SERVER = os.getenv("SQL_SERVER_IP")
    PORT= os.getenv("SQL_SERVER_PORT")
    DATABASE = os.getenv("DB")
    USER = os.getenv("DB_USER")
    PASSWORD = os.getenv("DB_PW")
    DRIVER = os.getenv("DB_DRIVER")
    engine = create_engine('mssql+pyodbc://{}:{}@{}:{}/{}?driver={}&TrustServerCertificate=yes'.format(USER, PASSWORD, SERVER, PORT, DATABASE, DRIVER))
    return engine