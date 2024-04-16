import pandas as pd

def split_X_y_z(table:pd.DataFrame):
    X = table.drop(columns=["Risk Level", "Class", "datapoint_id"]) #TODO:LoadFromDB
    y = pd.DataFrame(table["Risk Level"])
    z = pd.DataFrame(table["datapoint_id"])
    return X, y, z