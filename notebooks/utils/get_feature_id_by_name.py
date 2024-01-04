import pandas as pd

def get_feature_id_by_name(context, feature_name):
    """
    Get the ID of a feature by its name from the 'feature' table.

    Args:
        feature_name: The name of the feature.
    """

    session = context["session"]
    engine = session.get_bind()
    base = context["base"]
    query = session.query(base.classes.feature)
    df = pd.read_sql_query(query.statement, engine, chunksize=5)
    feature_id = df[df["name"] == feature_name].iloc[0]["id"]
    return feature_id
        