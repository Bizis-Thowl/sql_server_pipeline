import pandas as pd

class TrainInterface:
    def __init__(self, mlContext):
        self.mlContext = mlContext
        
    def get_model(self, i, args): # Derive Hyperopt args from method not from context.iter_args[i]!!!!!
        pass
    
    def populate(self, i):
        pass
    
    def upload(self, i):
        df = pd.DataFrame()
        df["pred"] = self.mlContext.iter_objs[i]["dtc_pred"]
        df["datapoint_id"] = self.mlContext.test_db_indexes
        df["model_id"] = self.mlContext.iter_objs[i]["model"]["dtc"].id
        df.to_sql("prediciions_categorical", con = self.mlContext.engine, index = False, if_exists='append')