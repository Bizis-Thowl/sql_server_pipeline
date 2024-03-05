from ..TrainPreperationInterface import TrainPreperationInterface
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.feature_selection import VarianceThreshold
from ...util import create_object, get_feature_id_by_name, get_next_ID_for_Table, update_object_attributes
import numpy as np
from ..defines import defines

class CorrelationFilter(TrainPreperationInterface):
    correlation_filter_threshold = "correlation_filter_threshold"
    
    def __init__(self, mlContext):
        super().__init__(mlContext)
        
    def calculate(self, i, args):
        #get vars
        threshold = args[CorrelationFilter.correlation_filter_threshold]
        #calculate
        highly_correlated_features = set()
        correlation_matrix = self.mlContext.iter_train_X[i].corr()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    colname = correlation_matrix.columns[i]
                    highly_correlated_features.update(colname)
        self.mlContext.iter_train_X[i].drop(columns = highly_correlated_features)        
        #db stuff
        update_object_attributes(context=self.mlContext, self.mlContext.iter_obs[i][defines.hyperparameter], commit=False
                                 correlation_filter_threshold = threshold)
        
        
    def populate(self, i):
        self.mlContext.iter_args[i][CorrelationFilter.correlation_filter_threshold] = 0.01 *  hp.randint(CorrelationFilter.correlation_filter_threshold, 15)
        
    def upload(self, i):
        pass
        