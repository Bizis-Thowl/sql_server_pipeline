import pandas as pd

def split_X_y_z(table:pd.DataFrame):
    y = pd.DataFrame(table["Risk Level"])
    z = table["datapoint_id"]
    X = table.drop(columns=["Risk Level", "datapoint_id"]) #TODO:LoadFromDB
    return X, y, z