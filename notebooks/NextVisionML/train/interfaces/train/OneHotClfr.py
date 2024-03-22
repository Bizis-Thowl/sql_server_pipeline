import torch
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pandas as pd
from ...TrainInterface import TrainInterface
from ....util import update_object_attributes, create_objects
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from skorch import NeuralNetClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from ...defines import defines
import pickle
import os

class OneHotClfr(TrainInterface):
    eval_predict = None
    
    def __init__(self, mlContext):
        super().__init__(mlContext)
        
    def get_model(self, i, args):
        args["num_classes"] = len(pd.unique(self.mlContext.train_y[self.mlContext.train_y.columns.values[0]]))
        self.args = args
        return OneHotClfrCustom(args)
    
    def upload(self, i):
        TrainInterface.upload(self, i)
        
        self.mlContext.iter_objs[i][defines.hyperparameter] = update_object_attributes(context = self.mlContext.context, entity = self.mlContext.iter_objs[i][defines.hyperparameter], commit = True,
            random_seed = int(self.args["random_seed"]),
            layer_count = int(self.args["layer_count"]),
            num_epochs = int(self.args["num_epochs"]),
        )
        
    def populate(self, i):
        args = {
            "random_seed": hp.randint("random_seed", 5000),
            "layer_count": hp.randint("layer_count", 2), 
            "num_epochs": hp.randint("num_epochs", 15),
        }
        self.mlContext.iter_args[i].update(args)
        
    def save_model(self, i):
        model_id = str(self.mlContext.iter_objs[i][defines.model].id)
        file_name = self.mlContext.iter_objs[i][defines.model].algorithm
        base_path = os.getenv("model_path")
        net_path = base_path + model_id + file_name
        with open(net_path, 'wb') as f:
            pickle.dump(self.model, f)
                 
class SimpleNetSoftmax(torch.nn.Module):
    def __init__(self, input_size, output_size, layer_count):
        self.input_size = input_size
        self.output_size = output_size
        self.layer_count = layer_count
        super(SimpleNetSoftmax, self).__init__()
        self.call_super_init
        self.fc = torch.nn.ModuleList()
        for _ in range(layer_count-1):
            self.fc.append(torch.nn.Linear(input_size, input_size*2))
            input_size*=2
        self.fc.append(torch.nn.Linear(input_size, output_size))

    def forward(self, X):
        y = X
        for layer in self.fc:
            y = layer(y)
        y = torch.softmax(y, dim=1)
        return y
    
    def __getstate__(self):        
        return self.state_dict()

    def __setstate__(self, state):
        self.load_state_dict(state)

class OneHotClfrCustom(BaseEstimator, TransformerMixin):  
    def __init__(self, args):
        self.args = args
        self.num_classes = args["num_classes"]
    
    def fit(self, X, y=None):
        input_size = len(X.columns)
        self.random_seed = self.args["random_seed"]
        self.layer_count = self.args["layer_count"]+2
        self.num_epochs = self.args["num_epochs"]
        
        train_X = torch.tensor(X.astype('float32').values)
        min, _ = train_X.min(dim=0)
        max, _ = train_X.max(dim=0)       
        normalized_X = (train_X - min) / (max-min + 1)
        normalized_X = normalized_X
        
        self.le = LabelEncoder()
        y = self.le.fit_transform(y.values.ravel())
        train_y = torch.tensor(y)
        
        one_hot_encoded = torch.nn.functional.one_hot(train_y.to(torch.int64), self.num_classes).float()
        torch.manual_seed(self.random_seed)
        self.net = SimpleNetSoftmax(input_size, self.num_classes, self.layer_count)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.005)    

        #Training loop
        for _ in range(self.num_epochs):
            outputs = self.net(normalized_X)
            loss = criterion(outputs, one_hot_encoded)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return self
    
    def predict(self, X):
        X = torch.tensor(X.astype('float32').values)
        min, _ = X.min(dim=0)
        max, _ = X.max(dim=0)       
        normalized_X = (X - min) / (max-min + 1)
        normalized_X = normalized_X  
        
        pred = self.net(normalized_X)
        _, pred = torch.max(pred, 1)
        pred = self.le.inverse_transform(pred)
        return pred

    def __getstate__(self):
        state = {}
        state["random_seed"] = self.random_seed
        state["layer_count"] = self.layer_count
        state["num_epochs"] = self.num_epochs
        state["le"] = pickle.dumps(self.le)
        state["net"] = self.net.state_dict()
        return

    def __setstate__(self, state):
        self.random_seed = state["random_seed"]
        self.layer_count = state["layer_count"] 
        self.num_epochs = state["num_epochs"]  
        self.le = pickle.loads(state["le"])
        net=SimpleNetSoftmax()
        self.net = net.load_state_dict(state["net"])

