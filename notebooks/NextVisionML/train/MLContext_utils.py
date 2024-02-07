import pandas as pd

def split_X_and_y(table:pd.DataFrame):
    X = table.drop(columns=["Risk Level", "Class"]) #TODO:LoadFromDB
    y = pd.DataFrame(table["Risk Level"])
    return X, y