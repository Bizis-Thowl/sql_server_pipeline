from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pandas as pd
from ...TrainInterface import TrainInterface
from ....util import update_object_attributes
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from defines import defines

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
        dtc = DecisionTreeClassifier(
            max_depth = args["max_depth"],
            min_samples_leaf = args["min_samples_leaf"],
            random_state = args["random_state"],
            max_features = args["max_features"],
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
        self.mlContext.iter_objs[i][defines.model] = update_object_attributes(self.mlContext.context, self.mlContext.iter_objs[i]["model"],
            algorithm = defines.decision_tree_classifier)
        
        
    def upload(self, i):
        super().upload()
        
        self.mlContext.iter_objs[i][defines.hyperparameter] = update_object_attributes(self.mlContext.context, self.mlContext.iter_objs[i][defines.hyperparameter], 
            max_depth = int(args["max_depth"]),
            min_samples_leaf = int(args["min_samples_leaf"]),
            random_state = int(args["random_state"]),
            max_features = int(args["max_features"]),
            criterion = args["criterion"],    
        )


        
    