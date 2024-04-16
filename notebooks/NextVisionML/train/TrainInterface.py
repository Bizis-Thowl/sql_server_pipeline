import pandas as pd
from .MLContext import MLContext
from .defines import defines
from ..util import update_object_attributes, create_object, get_next_ID_for_Table

class TrainInterface:
    def __init__(self, mlContext:MLContext):
        self.mlContext = mlContext
        
    def save_model(self, i):
        pass
        
    def get_model(self, i, args): # Derive Hyperopt args from method not from context.iter_args[i]!!!!!
        pass
    
    def populate(self, i):
        pass
    
    def upload(self, i):        
        for train_preparation in self.mlContext.train_preparation_methods:
            train_preparation.upload(i)

        self.mlContext.iter_objs[i]["model_score"] = update_object_attributes(context = self.mlContext, entity = self.mlContext.iter_objs[i]["model_score"], commit = True,
            balanced_accuracy_score  = self.balanced_accuracy_score)
        
        self.mlContext.iter_objs[i][defines.model] = update_object_attributes(context = self.mlContext, entity = self.mlContext.iter_objs[i][defines.model], commit = True,
            algorithm  = "model_" + type(self).__name__ + "_" + str(i) + ".model")
        
        df = pd.DataFrame()
        df["label_categorical_id"] = self.eval_predict

        df[df["label_categorical_id"] == "low"] == 1
        df[df["label_categorical_id"] == "low-med"] == 2
        df[df["label_categorical_id"] == "medium"] == 3
        df[df["label_categorical_id"] == "med-high"] == 4
        df[df["label_categorical_id"] == "high"] == 5
        
        test2 = self.mlContext.test_db_indexes.to_list()
        test3 = map(int, test2)
        test4 = list(test3)
        df["datapoint_id"] = test4
        df["model_id"] = self.mlContext.iter_objs[i]["model"].id
        df.apply(upload_pred, args = {self},  axis=1)
        self.mlContext.session.commit()
        self.save_model(i)

def upload_pred(row, self):
    test = row["datapoint_id"]
    datapoint = create_object(self.mlContext.context, "prediciions_categorical", with_commit=False,
        datapoint_id  = int(test),
        model_id = int(row["model_id"]),
        label_categorical_id =  int(row["label_categorical_id"]))
        
    self.mlContext.session.add(datapoint)