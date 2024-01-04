import os
from urllib.parse import quote

from dotenv import load_dotenv
from sqlalchemy import create_engine


def get_engine():
    load_dotenv()
    SERVER = quote(os.getenv("SQL_SERVER_IP"))
    PORT = quote(os.getenv("SQL_SERVER_PORT"))
    DATABASE = quote(os.getenv("DB"))
    USER = quote(os.getenv("DB_USER"))
    PASSWORD = quote(os.getenv("DB_PW"))
    DRIVER = quote(os.getenv("DB_DRIVER"))
    engine = create_engine(
        f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER}:{PORT}/{DATABASE}?driver={DRIVER}&TrustServerCertificate=yes")
    return engine
