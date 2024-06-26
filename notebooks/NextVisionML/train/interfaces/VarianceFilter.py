from ..TrainPreperationInterface import TrainPreperationInterface
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.feature_selection import _variance_threshold
from ...util import create_object, get_feature_id_by_name, get_next_ID_for_Table, update_object_attributes
import numpy as np
from ..defines import defines
from sklearn.feature_selection import VarianceThreshold

class VarianceFilter(TrainPreperationInterface):
    low_variant_signals = None
    mlContext = None
    
    def __init__(self, mlContext):
        super().__init__(mlContext)
        
    def calculate(self, i, args):
        #Calculate
        variance_threshold_floor = self.min_threshold_feature_variance
        variance_threshold_fac = self.max_threshold_feature_variance - self.min_threshold_feature_variance/100
        temp = variance_threshold_fac * args["variance_threshold_var_fac"]
        self.threshold =variance_threshold_floor + temp
              
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        cont_data = self.mlContext.iter_train_X[i].select_dtypes(include=numerics)
        selector = VarianceThreshold(threshold=self.threshold)
        selector.fit(cont_data)
        inverted_list = ~np.array(selector.get_support())   
        #Use a set to make sure that there are no duplicates
        low_variant_signals = set()
        low_variant_signals.update(cont_data.columns[inverted_list].tolist())
        
        self.mlContext.iter_train_X[i] = self.mlContext.train_X.drop(columns=list(low_variant_signals))
        self.mlContext.iter_test_X[i] = self.mlContext.test_X.drop(columns=list(low_variant_signals))
        
        self.low_variant_signals = low_variant_signals
    def populate(self, i):
        self.mlContext.iter_args[i]["variance_threshold_var_fac"] = hp.randint("variance_threshold", 100)        
        self.min_threshold_feature_variance = self.mlContext.init_parameters.min_threshold_feature_variance
        self.max_threshold_feature_variance = self.mlContext.init_parameters.max_threshold_feature_variance
        
        
    def upload(self, i):
        for _, signal in enumerate(self.low_variant_signals):
            create_object(self.mlContext.context, defines.dropped_feature_variance_filter, with_commit=True,
                id = int(get_next_ID_for_Table(self.mlContext.context, defines.dropped_feature_variance_filter)),
                train_process_iteration_compute_result_id = self.mlContext.iter_objs[i]["train_process_iteration_compute_result"].id,
                feature_id = int(get_feature_id_by_name(self.mlContext.context, signal)))
        update_object_attributes(context = self.mlContext, entity = self.mlContext.iter_objs[i][defines.hyperparameter], commit = True,
                                    variance_threshold = self.threshold)
        
        
        