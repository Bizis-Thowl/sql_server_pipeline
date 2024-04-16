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
    
    


    