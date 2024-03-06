
from ...TrainInterface import TrainInterface
from sklearn.decomposition import PCA
from hyperopt import hp
from ....util import update_object_attributes
from ...custom_pca import CustomPCA

class PcaUnsupervised(TrainInterface):
    
    def __init__(self, mlContext):
        super().__init__(mlContext)
        
    def get_model(self, i, args):        
        clf = CustomPCA(
            n_components = int(args["components"])
        )
        self.mlContext.iter_objs[i]["hyperparameter"] = update_object_attributes(self.mlContext.context, self.mlContext.iter_objs[i]["hyperparameter"], 
            n_components = int(args["components"])  
        )
        return clf

    def populate(self, i):
        num_dims = len(self.mlContext.iter_train_X[i])
        args = {
            "components": hp.choice('components', [(num_dims/5)*x for x in range(1, 5)]),
        }  
        self.mlContext.iter_args[i].update(args) 
        self.mlContext.iter_objs[i]["model"]["dtc"] = update_object_attributes(self.mlContext.context, self.mlContext.iter_objs[i]["model"]["dtc"],
                                                                        path_to_model = "dtc")

    def upload(self, i):
        return super().upload(i)

