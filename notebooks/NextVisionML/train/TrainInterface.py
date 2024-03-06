import pandas as pd
from .MLContext import MLContext
from .defines import defines
from ..util import update_object_attributes

class TrainInterface:
    def __init__(self, mlContext:MLContext):
        self.mlContext = mlContext
        
    def get_model(self, i, args): # Derive Hyperopt args from method not from context.iter_args[i]!!!!!
        pass
    
    def populate(self, i):
        pass
    
    def upload(self, i):
        for train_preparation in self.mlContext.train_methods:
            train_preparation.upload()
        
        update_object_attributes(context = self.mlContext, entitity = self.mlContext.iter_obs[i][defines.model], commit = False
                                 path_to_model  = "model_" + type(obj).__name__ + "_" + str(i) + ".model")
        
        df = pd.DataFrame()
        df["pred"] = self.mlContext.iter_objs[i]["dtc_pred"]
        df["datapoint_id"] = self.mlContext.test_db_indexes
        df["model_id"] = self.mlContext.iter_objs[i]["model"]["dtc"].id
        df.to_sql("prediciions_categorical", con = self.mlContext.engine, index = False, if_exists='append')