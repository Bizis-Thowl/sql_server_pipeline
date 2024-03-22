from ..TrainPreperationInterface import TrainPreperationInterface
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.feature_selection import VarianceThreshold
from ...util import create_object, get_feature_id_by_name, get_next_ID_for_Table, update_object_attributes
import numpy as np
from ..defines import defines

class CorrelationFilter(TrainPreperationInterface):
    class class_defines:
        correlation_filter_threshold = "correlation_filter_threshold"
        
    threshold = None
    
    def __init__(self, mlContext):
        super().__init__(mlContext)
        
    def calculate(self, i, args):
        #get vars
        self.threshold = args[CorrelationFilter.class_defines.correlation_filter_threshold]
        #calculate
        highly_correlated_features = list()
        correlation_matrix = self.mlContext.iter_train_X[i].corr()
        for k in range(len(correlation_matrix.columns)):
            for j in range(k + 1):
                if abs(correlation_matrix.iloc[k, j]) > self.threshold:
                    colname = correlation_matrix.columns[k]
                    highly_correlated_features.append(colname)
                    
        highly_correlated_features = list(set(highly_correlated_features))           
        self.mlContext.iter_train_X[i].drop(columns = highly_correlated_features)
        self.mlContext.iter_test_X[i].drop(columns = highly_correlated_features)
        
        self.dropped_signals = highly_correlated_features    
        
    def populate(self, i):
        self.mlContext.iter_args[i][CorrelationFilter.class_defines.correlation_filter_threshold] = 0.01 *  hp.randint(CorrelationFilter.class_defines.correlation_filter_threshold, 15)
        
    def upload(self, i):
        for _, signal in enumerate(self.dropped_signals):
            create_object(self.mlContext.context, defines.dropped_feature_variance_filter, with_commit=True,
                id = int(get_next_ID_for_Table(self.mlContext.context, defines.dropped_feature_variance_filter)),
                train_process_iteration_compute_result_id = self.mlContext.iter_objs[i]["train_process_iteration_compute_result"].id,
                feature_id = int(get_feature_id_by_name(self.mlContext.context, signal)))
        
        update_object_attributes(context = self.mlContext, entity = self.mlContext.iter_objs[i][defines.hyperparameter], commit = True,
                                    correlation_filter_threshold = self.threshold)
        