from ..TrainPreperationInterface import TrainPreperationInterface
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.feature_selection import VarianceThreshold
from ...util import create_object, get_feature_id_by_name, get_next_ID_for_Table
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from defines import defines 
from ...util import update_object_attributes

class CorrelationFilter(TrainPreperationInterface):
    class class_defines:
        mutual_info_num_cols_dropped = "mutual_info_num_cols_dropped"
    
    def __init__(self, mlContext):
        super().__init__(mlContext)
        
    def calculate(self, i, args):
        self.num_cols_dropped = args[self.class_defines.mutual_info_num_cols_dropped]
        output = []
        mutual_info = mutual_info_classif(self.mlContext.train_X[i], self.mlContext.train_y[i])
        order = np.argsort(mutual_info)
        sorted_cols = np.array(self.mlContext.train_X[i].columns)[order[::-1]]
        dropped_cols = sorted_cols[0:self.num_cols_dropped]
        sorted_cols
        return        
                       
    def populate(self, i):
        self.mlContext.iter_args[i][self.class_defines.mutual_info_num_cols_dropped] = hp.randint(self.class_defines.mutual_info_num_cols_dropped, 100)
        
    def upload(self, i):
        for _, signal in enumerate(self.low_variant_signals):
            create_object(self.mlContext.context, defines.dropped_feature_variance_filter, with_commit=True,
                id = int(get_next_ID_for_Table(self.mlContext.context, defines.dropped_feature_variance_filter)),
                train_process_iteration_compute_result_id = self.mlContext.iter_objs[i]["train_process_iteration_compute_result"].id,
                feature_id = int(get_feature_id_by_name(self.mlContext.context, signal)))
        
        update_object_attributes(context = self.mlContext, entity = self.mlContext.iter_objs[i][defines.hyperparameter], commit = True,
                                    num_cols_dropped = self.num_cols_dropped)
        
        