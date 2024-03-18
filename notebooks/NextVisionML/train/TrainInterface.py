import pandas as pd
from .MLContext import MLContext
from .defines import defines
from ..util import update_object_attributes, create_object, get_next_ID_for_Table

class TrainInterface:
    def __init__(self, mlContext:MLContext):
        self.mlContext = mlContext
        
    def get_model(self, i, args): # Derive Hyperopt args from method not from context.iter_args[i]!!!!!
        pass
    
    def populate(self, i):
        pass
    
    def upload(self, i):        
        for train_preparation in self.mlContext.train_preparation_methods:
            train_preparation.upload(i)
        
        update_object_attributes(context = self.mlContext, entity = self.mlContext.iter_objs[i][defines.model], commit = True,
            algorithm  = "model_" + type(self).__name__ + "_" + str(i) + ".model")
        
        df = pd.DataFrame()
        df["pred"] = self.eval_predict
        df["datapoint_id"] = self.mlContext.test_db_indexes
        df["model_id"] = self.mlContext.iter_objs[i]["model"].id
        df.to_sql("prediciions_categorical", con = self.mlContext.engine, index = False, if_exists='append')