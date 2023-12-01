-- Execute Python script using sp_execute_external_script
EXEC sp_execute_external_script
  @language = N'Python',
  @script = N'
import pandas as pd
import math
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from datetime import datetime, timedelta, timezone
import pyodbc
from sqlalchemy import Column, Date, Integer, String, Numeric, create_engine, Float, inspect, func, MetaData, Table, select, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import mapper, registry, Session, relationship
from sqlalchemy.ext.automap import automap_base
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error, balanced_accuracy_score 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import scipy
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

#SQLAlchemy Setup
def get_engine():
    SERVER = "localhost"
    DATABASE = "metmast"
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect=DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;TrustServerCertificate=yes;")
    return engine

engine = get_engine()
metadata = MetaData()
metadata.reflect(bind=engine)
Base = automap_base(metadata=metadata)
Base.prepare(autoload_with=engine)
mapper_registry = registry()

#########################################################################################################################################################################################
#data_meta
class data_meta:
    pass

# Create columns
columns = [ Column("id", Integer, primary_key=True)
            ]

# Create table
metadata = MetaData()
data_meta_table= Table("data_meta", metadata, *columns)
metadata.create_all(engine)

# Map the class imperatively
mapper_registry.map_imperatively(
    class_ = data_meta,
    local_table = data_meta_table
)
#########################################################################################################################################################################################
#feature
class feature:
    pass

# Create columns
columns = [ Column("id", Integer, primary_key=True),
            Column("data_meta_id", Integer, ForeignKey(data_meta.id)),
            Column("name", String),
            Column("type_", String),
            Column("description", String)
            ]

# Create table
metadata = MetaData()
feature_table= Table("feature", metadata, *columns)
metadata.create_all(engine)

# Map the class imperatively
mapper_registry.map_imperatively(
    class_ = feature,
    local_table = feature_table
)
#########################################################################################################################################################################################
#label
class label:
    pass

# Create columns
columns = [ Column("id", Integer, primary_key=True),
            Column("data_meta_id", Integer, ForeignKey(data_meta.id)),
            Column("name", String),
            Column("description", String)
            ]

# Create table
metadata = MetaData()
label_table= Table("label", metadata, *columns)
metadata.create_all(engine)

# Map the class imperatively
mapper_registry.map_imperatively(
    class_ = label,
    local_table = label_table
)
#########################################################################################################################################################################################
#label_categorical
class label_categorical:
    pass

# Create columns
columns = [ Column("id", Integer, primary_key=True),
            Column("label_id", Integer, ForeignKey(data_meta.id)),
            Column("category", String),
            Column("description", String)
            ]

# Create table
metadata = MetaData()
label_categorical_table= Table("label_categorical", metadata, *columns)
metadata.create_all(engine)

# Map the class imperatively
mapper_registry.map_imperatively(
    class_ = label_categorical,
    local_table = label_categorical_table
)
#########################################################################################################################################################################################
#datapoint_mappings
class datapoint_mappings:
    pass

# Create columns
columns = [ Column("id", Integer, primary_key=True),
            Column("data_meta_id", Integer, ForeignKey(data_meta.id)),
            Column("grouping", String)
            ]

# Create table
metadata = MetaData()
datapoint_mappings_table= Table("datapoint_mappings", metadata, *columns)
metadata.create_all(engine)

# Map the class imperatively
mapper_registry.map_imperatively(
    class_ = datapoint_mappings,
    local_table = datapoint_mappings_table
)
#########################################################################################################################################################################################
#datapoint
class datapoint:
    pass

# Create columns
columns = [ Column("id", Integer, primary_key=True),
            Column("datapoint_mappings_id", Integer, ForeignKey(datapoint_mappings.id))
            ]

# Create table
metadata = MetaData()
datapoint_table= Table("datapoint", metadata, *columns)
metadata.create_all(engine)

# Map the class imperatively
mapper_registry.map_imperatively(
    class_ = datapoint,
    local_table = datapoint_table
)
#########################################################################################################################################################################################
#datapoint_feature_value_int
class datapoint_feature_value_int:
    pass

# Create columns
columns = [ Column("id", Integer, primary_key=True),
            Column("datapoint_id", Integer, ForeignKey(datapoint.id)),
            Column("feature_id", Integer, ForeignKey(feature.id)),
            Column("int_", Integer)
            ]

# Create table
metadata = MetaData()
datapoint_feature_value_int_table= Table("datapoint_feature_value_int", metadata, *columns)
metadata.create_all(engine)

# Map the class imperatively
mapper_registry.map_imperatively(
    class_ = datapoint_feature_value_int,
    local_table = datapoint_feature_value_int_table
)
#########################################################################################################################################################################################
#datapoint_feature_value_float
class datapoint_feature_value_float:
    pass

# Create columns
columns = [ Column("id", Integer, primary_key=True),
            Column("datapoint_id", Integer, ForeignKey(datapoint.id)),
            Column("feature_id", Integer, ForeignKey(feature.id)),
            Column("float_", Float)
            ]

# Create table
metadata = MetaData()
datapoint_feature_value_float_table = Table("datapoint_feature_value_float", metadata, *columns)
metadata.create_all(engine)

# Map the class imperatively
mapper_registry.map_imperatively(
    class_ = datapoint_feature_value_float,
    local_table = datapoint_feature_value_float_table
)
#########################################################################################################################################################################################
#datapoint_feature_value_string
class datapoint_feature_value_string:
    pass

# Create columns
columns = [ Column("id", Integer, primary_key=True),
            Column("datapoint_id", Integer, ForeignKey(datapoint.id)),
            Column("feature_id", Integer, ForeignKey(feature.id)),
            Column("string_", String)
            ]

# Create table
metadata = MetaData()
datapoint_feature_value_string_table= Table("datapoint_feature_value_string", metadata, *columns)
metadata.create_all(engine)

# Map the class imperatively
mapper_registry.map_imperatively(
    class_ = datapoint_feature_value_string,
    local_table = datapoint_feature_value_string_table
)
#Loading Data
failures_2016 = pd.read_csv(r"C:\Raw Data\failures-2016.csv", sep=";")
failures_2017 = pd.read_csv(r"C:\Raw Data\failures-2017.csv", sep=";")
metmast_2016 = pd.read_csv(r"C:\Raw Data\metmast-2016.csv", sep=";")
metmast_2017 = pd.read_csv(r"C:\Raw Data\metmast-2017.csv", sep=";")
signals_2016 = pd.read_csv(r"C:\Raw Data\signals-2016.csv", sep=";")
signals_2017 = pd.read_csv(r"C:\Raw Data\signals-2017.csv", sep=";")

# Signale beider Jahre kombinieren
signals = pd.concat([signals_2016, signals_2017])

turbine_names = signals["Turbine_ID"].unique()

def create_df_for_each_turbine(signals):
    turbine_dfs = list();

    for turbine in turbine_names:
        turbine_df = signals[signals["Turbine_ID"] == turbine]
        turbine_df = turbine_df.sort_values("Timestamp")
        turbine_df = turbine_df.reset_index(drop=True)
        turbine_dfs.append(turbine_df)

    return turbine_dfs

turbine_dfs = create_df_for_each_turbine(signals)

#Zusammenführen und sortieren
metmast = pd.concat([metmast_2016, metmast_2017])
metmast = metmast.sort_values("Timestamp")

# drop broken met data
metmast = metmast.drop(["Min_Winddirection2", "Max_Winddirection2", "Avg_Winddirection2", "Var_Winddirection2"], axis=1)

# Fill met data
metmast = metmast.fillna(method = "ffill")
metmast = metmast.fillna(method = "bfill")
metmast.isna().sum().sum()

failures = pd.concat([failures_2016, failures_2017])

#Mergen
def JoinMetamast(df:pd.DataFrame):
    df = df.fillna(method = "ffill")
    df = df.fillna(method = "bfill")
    df = pd.merge(df, metmast, on="Timestamp", how="left")
    df = df.fillna(method = "ffill")
    df = df.fillna(method = "bfill")
    df.isna().sum().sum()
    return df

merged = list()
for turbine_df in turbine_dfs:
    merged.append(JoinMetamast(turbine_df))
merged_df = pd.concat(merged)

failures_gearbox = failures[failures["Component"] == "GEARBOX"]
failures_gearbox.reset_index(drop=True, inplace=True)

#Util Functions
def get_round_minute_diff(datetime_in: datetime) -> timedelta:
    min = datetime_in.minute
    rounded_min = round(min, -1)
    diff = rounded_min - min
    return timedelta(minutes=diff)

def convert_round_minute_to_time(datetime_in: datetime) -> datetime:
    td = get_round_minute_diff(datetime_in)
    return datetime_in + td

days_lookback = 90
mins_per_class = 24 * 60 / 10
ten_mins_of_n_days = int(24 * 60 * days_lookback / 10) 
target_name = "Class"

def GetClass(i:int)->int:
    return math.floor(i/mins_per_class)

def create_failure_list() -> pd.DataFrame:
    failure_list = []
    for i, failure in enumerate(failures_gearbox):
        turbine_id = str(failures_gearbox["Turbine_ID"][i])
        failure_ts = str(failures_gearbox["Timestamp"][i])
        failure_datetime = datetime.fromisoformat(failure_ts)
        rounded_datetime = convert_round_minute_to_time(failure_datetime)
        for j in range(ten_mins_of_n_days):
            delta = timedelta(minutes=j*10)
            new_datetime = rounded_datetime - delta
            datetime_formated = new_datetime.replace(tzinfo=timezone.utc)
            failure_list.append([turbine_id, datetime_formated.isoformat(), GetClass(j)])    
    failure_df = pd.DataFrame(failure_list, columns=["Turbine_ID", "Timestamp", target_name])
    return failure_df

failure_df_class  = create_failure_list()
#Der Feature-Datensatz wird mit den Labels zusammengeführt. Dabei ist besonders wichtig, dass der Bezug zu der jeweiligen Turbine bestehen bleibt.
labeled_df = pd.merge(merged_df, failure_df_class, on=["Turbine_ID", "Timestamp"], how="left");
labeled_df = labeled_df.reset_index(drop=True)

def create_failure_list(classes: list[str], days_per_class: int, target_name: str) -> pd.DataFrame:
    days_lookback = len(classes) * days_per_class
    ten_mins_of_n_days = int(24 * 60 * days_lookback / 10)
    failure_list = []
    for i, failure in enumerate(failures_gearbox):
        turbine_id = str(failures_gearbox["Turbine_ID"][i])
        failure_ts = str(failures_gearbox["Timestamp"][i])
        failure_datetime = datetime.fromisoformat(failure_ts)
        rounded_datetime = convert_round_minute_to_time(failure_datetime)
        for iterator, current_class in enumerate(classes):
            for j in range(ten_mins_of_n_days):
                delta = timedelta(minutes=j*10)
                # Prüfen ob obere und untere Schranke passen.
                is_in_class = delta >= timedelta(days=iterator*days_per_class) and delta < timedelta(days=(iterator+1) * days_per_class)
                if (is_in_class):
                    new_datetime = rounded_datetime - delta
                    datetime_formated = new_datetime.replace(tzinfo=timezone.utc)
                    failure_list.append([turbine_id, datetime_formated.isoformat(), current_class])
    
    failure_df = pd.DataFrame(failure_list, columns=["Turbine_ID", "Timestamp", target_name])

    return failure_df

class_target_name = "Risk Level"
risk_levels = ["low", "high", "med-high", "medium", "low-med"]
days_per_class = 18

failure_df_multiclass = create_failure_list(classes=risk_levels, days_per_class=days_per_class, target_name=class_target_name)
labeled_df = pd.merge(labeled_df, failure_df_multiclass, on=["Turbine_ID", "Timestamp"], how="left"); 
labeled_df = labeled_df.reset_index(drop=True)

labeled_df[class_target_name].fillna("low", inplace = True)
labeled_df[target_name].fillna(90, inplace = True)

labeled_df[target_name].value_counts()

# Alle Daten ab August 2017 liegen im Testset
split_criterion_reg = labeled_df["Timestamp"] >= "2017-08-00T00:00:00+00:00"

test_gearbox = labeled_df[split_criterion_reg].reset_index(drop=True)#.iloc[:100].reset_index(drop=True)
train_gearbox = labeled_df[~split_criterion_reg].reset_index(drop=True)#.iloc[:100].reset_index(drop=True)

#Helper functions
def getNextIdForTable(context, table_name):
    table = context["base"].classes[table_name]
    return context["session"].query(table.id).count() + 1

def create_object(context, table_name, **kwargs):
    mapped_class = context["base"].classes[table_name]
    obj = mapped_class()
    #setattr(obj, "id", getNextIdForTable(context, table_name))
    for key, value in kwargs.items():
        setattr(obj, key, value)
    context["session"].add(obj)
    context["session"].commit()
    return obj

def update_values(obj, ref_class, session, **kwargs):
    #clone = ref_class()
    for column in ref_class.__table__.columns:
        if(column.name in kwargs):
            setattr(obj, column.name, kwargs[column.name])
        #else:
            #setattr(clone, column.name, getattr(obj, column.name))
    #session.delete(obj)
    #session.add(clone)
    session.commit() 
    return obj
def get_feature_id_by_name(Base, name):
    query = session.query(metadata.tables["feature"])
    df = pd.DataFrame()
    for batch in pd.read_sql_query(query.statement, engine, chunksize= 5):
        df = pd.concat([df, batch], ignore_index=True)
    for index, row in df.iterrows():
        if(row["name"]==name):
            return row["id"]
  
#SQLAlchemy Setup
def get_engine():
    SERVER = "localhost"
    DATABASE = "metmast"
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect=DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;TrustServerCertificate=yes;")
    return engine

engine = get_engine()
metadata = MetaData()
metadata.reflect(bind=engine)
Base = automap_base(metadata=metadata)
Base.prepare(autoload_with=engine)
mapper_registry = registry()

context = dict()
context["base"] = Base
context["session"] = Session(bind=engine)

if(getNextIdForTable(context, "data_meta")==1):
    data_meta = create_object(context, "data_meta")

groupings = ["train", "test"]
grouping_ids = dict()
if(getNextIdForTable(context, "datapoint_mappings")<len(groupings)):
    for i, grouping_val in enumerate(groupings):
        create_object(context, "datapoint_mappings",
            data_meta_id = 1,
            grouping = grouping_val
        )
for i, grouping_val in enumerate(groupings):
    grouping_ids[grouping_val] = i + 1

label_names = ["Class", "Risk Level"]
label_ids = dict()
if(getNextIdForTable(context, "label")<len(label_names)):
    for i, label in enumerate(label_names):
        create_object(context, "label",
            data_meta_id = 1,
            name = label,
            description = "TBD"
        )
for i, name in enumerate(label_names):
    label_ids[name] = i + 1

meta_info_names = ["Turbine_ID", "Timestamp"]
feature_names = [feature for feature in train_gearbox.columns if feature not in label_names and feature not in meta_info_names]
aggregated_meta_feature_list = meta_info_names + feature_names
aggregated_meta_feature_list = [feature for feature in aggregated_meta_feature_list if feature!="internal_string_102412"]
feature_ids = dict()
meta_info_ids = dict()
if(getNextIdForTable(context, "feature")<len(aggregated_meta_feature_list)):
    for i, name in enumerate(aggregated_meta_feature_list):
        type_ = "data"
        if name in meta_info_names:
            type_ = "meta"
        create_object(context, "feature",
            data_meta_id = 1,
            name = name,
            type_ = type_,
            description = "TBD"
        )
for i, name in enumerate(aggregated_meta_feature_list):
    if name in meta_info_ids:
        meta_info_ids[name] = i + 1
    else:
        feature_ids[name] = i + 1
aggregated_meta_feature_ids = {**feature_ids, **meta_info_ids}

train_gearbox["internal_string_102412"] = "train"
test_gearbox["internal_string_102412"] = "test"
data = pd.concat([train_gearbox, test_gearbox], axis = 0)
helper_string_row_index = data.columns.get_loc("internal_string_102412")
if(getNextIdForTable(context, "datapoint")<2):    
    for i, row in enumerate(data.iterrows()):
        _, row = row  
        datapoint = create_object(context, "datapoint",
            datapoint_mappings_id = grouping_ids[row[helper_string_row_index]]
        )
        for key, value in row.items():
            if(key not in label_names and key!="internal_string_102412"):
                if "float" in str(type(value)):
                    create_object(context, "datapoint_feature_value_float",
                        datapoint_id = datapoint.id,
                        feature_id = aggregated_meta_feature_ids[key],
                        float_ = float(value)
                    )
                elif "int" in str(type(value)):
                    create_object(context, "datapoint_feature_value_int",
                        datapoint_id = datapoint.id,
                        feature_id = aggregated_meta_feature_ids[key],
                        int_ = int(value)
                    )
                else: #"string" in str(type(value)):
                    create_object(context, "datapoint_feature_value_string",
                        datapoint_id = datapoint.id,
                        feature_id = aggregated_meta_feature_ids[key],
                        string_ = str(value) 
                    )
';