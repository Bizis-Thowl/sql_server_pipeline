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
        
    def get_model(self, i, args):
        dtc = DecisionTreeClassifier(
            max_depth = args["max_depth"],
            min_samples_leaf = args["min_samples_leaf"],
            random_state = args["random_state"],
            max_features = args["max_features"],
            criterion = args["criterion"],
        )
        self.mlContext.iter_objs[i]["hyperparameter"] = update_object_attributes(self.mlContext.context, self.mlContext.iter_objs[i]["hyperparameter"], 
            max_depth = int(args["max_depth"]),
            min_samples_leaf = int(args["min_samples_leaf"]),
            random_state = int(args["random_state"]),
            max_features = int(args["max_features"]),
            criterion = args["criterion"],    
        )
        return dtc
        #dtc.fit(self.mlContext.iter_train_X[i], self.mlContext.iter_train_y[i])
        #self.mlContext.iter_objs[i]["model"]["dtc"] = dtc
        #eval_predict = dtc.predict(self.mlContext.iter_test_X[i]) 
        #balanced_accuracy = balanced_accuracy_score(self.mlContext.iter_test_y[i], eval_predict) 
        
        #self.mlContext.iter_objs[i]["dtc_pred"] = eval_predict
        #self.mlContext.iter_objs[i]["dtc_balanced_accuracy"] = balanced_accuracy
        
        test_pred = pd.DataFrame(eval_predict.reshape(-1, 1))  
        class_mapping = {"low": 0, "low-med": 1, "medium": 2, "med-high": 3, "high": 4}
        train_y_ = self.mlContext.iter_train_y[i]["Risk Level"].map(class_mapping)
        test_pred = test_pred[0].map(class_mapping)
        self.mlContext.iter_objs[i]["dtc_trust_scores"] = get_trust_scores(self.mlContext.iter_train_X[i].values, train_y_, self.mlContext.iter_test_X[i].values, test_pred)
        
    
        
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
    def upload(self, i):
        super().upload()
        self.mlContext.iter_objs[i]["model_score_dtc"] = update_object_attributes(self.mlContext.context, self.mlContext.iter_objs[i]["model_score_dtc"],
            balanced_accuracy_score = self.mlContext.iter_objs[i]["dtc_balanced_accuracy"])
        
        df = pd.DataFrame()        
        df["pred"] = self.mlContext.iter_objs[i]["dtc_pred"]
        df["trust_score"] = self.mlContext.iter_objs[i]["dtc_trust_scores"]
        df["datapoint_id"] = self.mlContext.test_db_indexes
        df["model_id"] = self.mlContext.iter_objs[i]["model"]["dtc"].id
        df.to_sql("prediciions_categorical", con = self.mlContext.engine, index = False, if_exists='append')

        
    