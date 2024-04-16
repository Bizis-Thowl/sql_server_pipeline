
from ...TrainInterface import TrainInterface
from sklearn.decomposition import PCA
from hyperopt import hp
from ....util import update_object_attributes
from ...custom_pca import CustomPCA
from ...defines import defines
import pickle
import os

class PcaUnsupervised(TrainInterface):
    
    def __init__(self, mlContext):
        super().__init__(mlContext)
        
    def get_model(self, i, args):
        self.args = args    
        
        clf = CustomPCA(
            n_components = int(args["components"])
        )
        
        return clf

    def populate(self, i):
        args = {
            "components": hp.choice('components', [10, 30, 50, 70]),
        }  
        self.mlContext.iter_args[i].update(args)

    def upload(self, i):
        TrainInterface.upload(self, i)
        
        self.mlContext.iter_objs[i][defines.hyperparameter] = update_object_attributes(context = self.mlContext.context, entity = self.mlContext.iter_objs[i][defines.hyperparameter], commit = True,
            components = int(self.args["components"])
        )
    
    def save_model(self, i):
        model_id = str(self.mlContext.iter_objs[i][defines.model].id)
        file_name = self.mlContext.iter_objs[i][defines.model].algorithm
        base_path = os.getenv("model_path")
        net_path = base_path + model_id + file_name
        with open(net_path, 'wb') as f:
            pickle.dump(self.model, f)

