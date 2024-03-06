import pandas as pd
import os
from urllib.parse import quote
from dotenv import load_dotenv
from sqlalchemy import create_engine

def get_engine():
    load_dotenv()
    
    if(os.getenv("trusted")=="yes"):
        return get_engine_trusted()
    
    SERVER = os.getenv("SQL_SERVER_IP")
    PORT= os.getenv("SQL_SERVER_PORT")
    DATABASE = os.getenv("DB")
    USER = os.getenv("DB_USER")
    PASSWORD = os.getenv("DB_PW")
    DRIVER = os.getenv("DB_DRIVER")
    engine = create_engine(
        f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER}:{PORT}/{DATABASE}?driver={DRIVER}&TrustServerCertificate=yes", pool_size=10000, max_overflow=-1)
    return engine

def get_engine_trusted():
    server = 'localhost'
    database = 'metmast_0_4'

    # Construct the connection string with trusted connection
    connection_string = f'mssql+pyodbc://@{server}:1433/{database}?trusted_connection=yes&driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes' #TODO:Certificate
    engine = create_engine(connection_string, pool_size=1000, max_overflow=-1)
    return engine

def create_object(context, table_name, with_commit=False, **kwargs):
    mapped_class = context["Base"].classes[table_name]
    obj = mapped_class()
    for key, value in kwargs.items():
        setattr(obj, key, value)
    context["session"].add(obj)
    if (with_commit):
        context["session"].commit()
    return obj

def create_objects(context, table_name, object_data_list):
    """
    Create multiple objects and add them to the database in a batch.

    :param context: The database context.
    :param table_name: The name of the table where objects will be created.
    :param object_data_list: A list of dictionaries, where each dictionary contains the attributes for one object.
    :return: A list of created objects.
    """
    mapped_class = context["Base"].classes[table_name]
    objects = []

    for data in object_data_list:
        obj = mapped_class()
        for key, value in data.items():
            setattr(obj, key, value)
        context["session"].add(obj)
        objects.append(obj)

    return objects

#TODO Needs redesign for new database
def get_feature_id_by_name(context, feature_name):
    """
    Get the ID of a feature by its name from the 'feature' table.

    Args:
        feature_name: The name of the feature.
    """

    session = context["session"]
    engine = session.get_bind()
    base = context["Base"]
    query = session.query(base.classes.feature)
    df = pd.read_sql_query(query.statement, engine.connect())
    feature_id = df[df["name"] == feature_name].iloc[0]["id"]
    return feature_id

def get_next_ID_for_Table(sqlContext:dict, name:str):
    table = sqlContext["Base"].classes[name]
    return sqlContext["session"].query(table.id).count() + 1 
        
def update_object_attributes(context, entity, commit = True, **kwargs):
    """
    Update the attributes of a given SQLAlchemy entity.

    Args:
        entity: The SQLAlchemy entity to be updated.
        attributes: A dictionary of attribute names and their new values.
    """
    for attr, value in kwargs.items():
        setattr(entity, attr, value)
    #context["session"].add(entity) necessary??
    if(commit): context["session"].commit()
    return entity