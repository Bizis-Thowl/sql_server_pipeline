import pickle
import pandas as pd

from hyperopt import hp
from ...TrainInterface import TrainInterface
from ....util import update_object_attributes, create_object, get_next_ID_for_Table
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from ...defines import defines
import os

class DecisionTreeClfr(TrainInterface):
    class class_defines:
        max_depth = "max_depth"
        min_samples_leaf  = "min_samples_leaf"
        random_state = "random_state"
        max_features = "max_features"
        criterion = "criterion"
        
    def __init__(self, mlContext):
        super().__init__(mlContext)
        
    def get_model(self, i, args):
        extracted_dict = {
            "max_depth": args["max_depth"],
            "min_samples_leaf": args["min_samples_leaf"],
            "random_state": args["random_state"],
            "max_features": args["max_features"],
            "criterion": args["criterion"]
        }
        self.args = extracted_dict
        dtc = DecisionTreeClassifier(
            max_depth = extracted_dict["max_depth"],
            min_samples_leaf = extracted_dict["min_samples_leaf"],
            random_state = extracted_dict["random_state"],
            max_features = extracted_dict["max_features"],
            criterion = args["criterion"],
        )        
        return dtc             
    
        
    def populate(self, i):
        args = {
            "max_depth": hp.choice('max_depth', range(1,100)),
            "min_samples_leaf": hp.choice("min_samples_leaf", range(1,15)),
            "random_state": hp.randint("random_state", 3000),
            "max_features": hp.choice('max_features', range(1,50)),
            "criterion": hp.choice('criterion', ["gini", "entropy"]),
        }  
        self.mlContext.iter_args[i].update(args) 
        self.mlContext.iter_objs[i][defines.model] = update_object_attributes(context = self.mlContext, entity = self.mlContext.iter_objs[i]["model"], with_commit=True,
            algorithm = defines.decision_tree_classifier)
        
        
    def upload(self, i):
        TrainInterface.upload(self, i)
        
        self.mlContext.iter_objs[i][defines.hyperparameter] = update_object_attributes(context = self.mlContext.context, entity = self.mlContext.iter_objs[i][defines.hyperparameter], commit = True,
            max_depth = int(self.args["max_depth"]),
            min_samples_leaf = int(self.args["min_samples_leaf"]),
            random_state = int(self.args["random_state"]),
            max_features = int(self.args["max_features"]),
            criterion = self.args["criterion"],    
        )  
        
    def save_model(self, i):
        model_id = str(self.mlContext.iter_objs[i][defines.model].id)
        file_name = self.mlContext.iter_objs[i][defines.model].algorithm
        base_path = os.getenv("model_path")
        dtr_path = base_path + model_id + file_name
        with open(dtr_path, 'wb') as f:
            pickle.dump(self.model, f)