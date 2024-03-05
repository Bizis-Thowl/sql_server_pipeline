from ..TrainPreperationInterface import TrainPreperationInterface
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.feature_selection import VarianceThreshold
from ...util import create_object, get_feature_id_by_name, get_next_ID_for_Table
import numpy as np
from sklearn.feature_selection import mutual_info_classif

class CorrelationFilter(TrainPreperationInterface):
    def __init__(self, mlContext):
        super().__init__(mlContext)
        
    def calculate(self, i, args):
        output = []
        mutual_info = mutual_info_classif(X_train, y_train)
        order = np.argsort(mutual_info)
        sorted_cols = np.array(X_train.columns)[order[::-1]]
        dropped_cols = sorted_cols[0:num_cols]
        sorted_cols
        return output        
                       
    def populate(self, i):
        self.mlContext.iter_args[i]["variance_threshold_var_fac"] = hp.randint("variance_threshold", 100)
        
    def upload(self, i):
        pass
        
        