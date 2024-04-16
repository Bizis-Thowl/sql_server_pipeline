import pandas as pd
from sqlalchemy import MetaData, select
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session, registry
from ..util import get_engine

def load_context():    
    """
    Creates an SQL Context for sql sqlalchemy and loads the data from db.
    
    Returns:
        tuple:
            dict: A dictionary containing a bunch of sql related objects.
            DataFrame: A table with features and labels from the db´for training
            DataFrame: A table with features and labels from the db for testing
    """
    engine = get_engine()
    metadata = MetaData()
    metadata.reflect(bind=engine)
    Base = automap_base(metadata=metadata)
    Base.prepare(autoload_with=engine)    
    mapper_registry = registry()
    
    sqlContext = dict()
    sqlContext["metadata"] = metadata
    sqlContext["Base"] = Base
    sqlContext["engine"] = engine  
    con = engine.connect()    
    
    datapoint = pd.read_sql_table("datapoint", con)
    #labels = pd.read_sql_table("label", con)
    #label_values = pd.read_sql_table("label_categorical", con)
    datapoint_label_values = pd.read_sql_table("datapoint_class_label", con)
    features = pd.read_sql_table("feature", con)
    datapoint_features = pd.read_sql_table("datapoint_feature_value", con)
    datapoint = datapoint.rename(columns={'id': "datapoint_id"})

    for i, row in features.iterrows():
        feature_id = row["id"]
        feature_df = datapoint_features[datapoint_features["feature_id"]==feature_id]
        feature_df = feature_df.drop(columns={"id", "feature_id"})
        datapoint = pd.merge(datapoint, feature_df, on='datapoint_id')
        datapoint.rename(columns={'value': row["name"]}, inplace=True)
        if "datapoint_id" in datapoint.columns:
            pass
            #datapoint = datapoint.drop(columns={"datapoint_id"})
        if "id_x" in datapoint.columns:
            datapoint = datapoint.drop(columns={"id_x"})
        if "id_y" in datapoint.columns:
            datapoint = datapoint.drop(columns={"id_y"})

    datapoint = pd.merge(datapoint, datapoint_label_values, on='datapoint_id')
    datapoint = datapoint.rename(columns={'label_categorical_id': "Risk Level"})

    datapoint[datapoint["Risk Level"] == 1] == "low"
    datapoint[datapoint["Risk Level"] == 2] == "low-med"
    datapoint[datapoint["Risk Level"] == 3] == "medium"
    datapoint[datapoint["Risk Level"] == 4] == "med-high"
    datapoint[datapoint["Risk Level"] == 5] == "high"

    train = datapoint[datapoint["datapoint_mappings_id"]==1]
    test = datapoint[datapoint["datapoint_mappings_id"]==2]

    train = train.drop(columns={"datapoint_mappings_id", "datetime", "id"})
    test = test.drop(columns={"datapoint_mappings_id", "datetime", "id"})
    return sqlContext, train, test
    
      
    
    #Metadata
    datapoint = metadata.tables['datapoint']
    datapoint_feature_value = metadata.tables['datapoint_feature_value']
    feature = metadata.tables['feature']
    datapoint_mappings = metadata.tables['datapoint_mappings']
    datapoint_class_label = metadata.tables["datapoint_class_label"]
    datapoint_rul_label = metadata.tables["datapoint_rul_label"]
    label = metadata.tables["label"]    
    
    # For train set
    query_train = select(
        datapoint.c.id.label('datapoint_id'),
        feature.c.name.label('feature_name'),
        datapoint_feature_value.c.value.label('feature_value')
    ).select_from(
        datapoint.join(datapoint_feature_value, datapoint.c.id == datapoint_feature_value.c.datapoint_id)
        .join(feature, feature.c.id == datapoint_feature_value.c.feature_id)
        .join(datapoint_mappings, datapoint.c.datapoint_mappings_id == datapoint_mappings.c.id)
    ).where(
        datapoint_mappings.c.grouping == 'train'
    )

    # For test set
    query_test = select(
        datapoint.c.id.label('datapoint_id'),
        feature.c.name.label('feature_name'),
        datapoint_feature_value.c.value.label('feature_value')
    ).select_from(
        datapoint.join(datapoint_feature_value, datapoint.c.id == datapoint_feature_value.c.datapoint_id)
        .join(feature, feature.c.id == datapoint_feature_value.c.feature_id)
        .join(datapoint_mappings, datapoint.c.datapoint_mappings_id == datapoint_mappings.c.id)
    ).where(
        datapoint_mappings.c.grouping == 'test'
    )
    
    # Construct SQL query for categorical labels
    query_cat_labels = select(
        datapoint.c.id.label('datapoint_id'),
        label.c.name.label('label_name'),
        datapoint_class_label.c.value.label('label_value')
    ).select_from(
        datapoint.join(datapoint_class_label, datapoint.c.id == datapoint_class_label.c.datapoint_id)
        .join(label, label.c.id == datapoint_class_label.c.label_id)
    )

    # Construct SQL query for continuous labels
    query_cont_labels = select(
        datapoint.c.id.label('datapoint_id'),
        label.c.name.label('label_name'),
        datapoint_rul_label.c.value.label('label_value')
    ).select_from(
        datapoint.join(datapoint_rul_label, datapoint.c.id == datapoint_rul_label.c.datapoint_id)
        .join(label, label.c.id == datapoint_rul_label.c.label_id)
    )
    
    con = engine.connect()
    df_train = pd.read_sql_query(query_train, con)
    df_test = pd.read_sql_query(query_test, con)

    # Pivot tables to get features as columns, for both train and test
    train = df_train.pivot_table(index='datapoint_id', columns='feature_name', values='feature_value').reset_index()
    test = df_test.pivot_table(index='datapoint_id', columns='feature_name', values='feature_value').reset_index()
    
    # Execute the queries and load into DataFrames
    df_cat_labels = pd.read_sql_query(query_cat_labels, con)
    df_cont_labels = pd.read_sql_query(query_cont_labels, con)

    # Pivot table for categorical labels
    pivot_cat_labels = df_cat_labels.pivot_table(
        index='datapoint_id',
        columns='label_name',
        values='label_value',
        aggfunc='first'  # Use 'first' or 'max' for categorical data
    ).reset_index()

    # Pivot table for continuous labels
    pivot_cont_labels = df_cont_labels.pivot_table(
        index='datapoint_id',
        columns='label_name',
        values='label_value',
        aggfunc='mean'  # Use 'mean' or another suitable function for numerical data
    ).reset_index()
    
    # Merge categorical and continuous labels
    pivot_labels = pd.merge(pivot_cat_labels, pivot_cont_labels, on='datapoint_id', how='outer')

    # Merge labels with feature dataframes
    train_with_labels = pd.merge(train, pivot_labels, on='datapoint_id', how='left')
    test_with_labels = pd.merge(test, pivot_labels, on='datapoint_id', how='left')

    return sqlContext, train_with_labels, test_with_labels
    
    


    