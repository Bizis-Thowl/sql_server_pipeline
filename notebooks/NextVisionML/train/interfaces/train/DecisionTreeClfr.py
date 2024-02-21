from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pandas as pd
from ...TrainInterface import TrainInterface
from ....util import update_object_attributes
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from .temp_util import get_trust_scores

class DecisionTreeClfr(TrainInterface):
    def __init__(self, mlContext):
        super().__init__(mlContext)
        
    def calculate(self, i, args):
        update_object_attributes(self.mlContext.context, self.mlContext.iter_objs[i]["hyperparameter"], 
                    max_depth = int(args["max_depth"]),
                    min_samples_leaf = int(args["min_samples_leaf"]),
                    random_state = int(args["random_state"]),
                    max_features = int(args["max_features"]),
                    criterion = args["criterion"],    
        ) 
        dtc = DecisionTreeClassifier(
            max_depth = args["max_depth"],
            min_samples_leaf = args["min_samples_leaf"],
            random_state = args["random_state"],
            max_features = args["max_features"],
            criterion = args["criterion"],
        )
        dtc.fit(self.mlContext.iter_train_X[i], self.mlContext.iter_train_y[i])
        self.mlContext.iter_objs[i]["model"]["dtc"] = dtc
        eval_predict = dtc.predict(self.mlContext.iter_test_X[i]) 
        balanced_accuracy = balanced_accuracy_score(self.mlContext.iter_test_y[i], eval_predict) 
        
        self.mlContext.iter_objs[i]["dtc_pred"] = eval_predict
        self.mlContext.iter_objs[i]["dtc_balanced_accuracy"] = balanced_accuracy
        
        return balanced_accuracy        
        
    def populate(self, i):
        args = {
            "max_depth": hp.choice('max_depth', range(1,100)),
            "min_samples_leaf": hp.choice("min_samples_leaf", range(1,15)),
            "random_state": hp.randint("random_state", 3000),
            "max_features": hp.choice('max_features', range(1,50)),
            "criterion": hp.choice('criterion', ["gini", "entropy"]),
        }  
        self.mlContext.iter_args[i].update(args) 
        self.mlContext.iter_objs[i]["model"]["dtc"] = update_object_attributes(self.mlContext.context, self.mlContext.iter_objs[i]["model"]["dtc"],
                                                                        path_to_model = "dtc")
    def update(self, i, args):
        self.mlContext.iter_objs[i]["model_score"] = update_object_attributes(self.mlContext.mlContext.context, self.mlContext.iter_objs[i]["model_score"],
            balanced_accuracy_score = self.mlContext.iter_objs[i]["dtc_balanced_accuracy"])
        self.mlContext.iter_objs[i]["model_score"] = update_object_attributes(self.mlContext.mlContext.context, self.mlContext.iter_objs[i]["hyperparameter"], 
            max_depth = int(args["max_depth"]),
            min_samples_leaf = int(args["min_samples_leaf"]),
            random_state = int(args["random_state"]),
            max_features = int(args["max_features"]),
            criterion = args["criterion"],    
        ) 
        
        
        df = pd.Dataframe()        
        df["pred"] = self.mlContext.iter_objs[i]["dtc_pred"]
        df["datapoint_id"] = self.mlContext.test_db_indexes
        df["model_id"] = self.mlContext.iter_objs[i]["model"].id
        df.reset_index(inplace=True)
        df.to_sql("metmast", con = self.mlContext.engine, index = False, if_exists='append')

        
    