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

class OneHotClfr(TrainInterface):
    eval_predict = None
    
    def __init__(self, mlContext):
        super().__init__(mlContext)
        
    def get_Model(self, i, args):
        return OneHotClfrCustom(mlcontext=self.mlContext, args = args)
    
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
                 
class SimpleNetSoftmax(torch.nn.Module):
    def __init__(self, input_size, output_size, layer_count):
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

class OneHotClfrCustom(BaseEstimator, TransformerMixin):  
  def __init__(self, mlcontext, args):
    self.mlcontext = mlcontext
    self.args = args
    
  def fit(self, X, y=None):
    input_size = len(X.columns)
    random_seed = self.args["random_seed"]    
    self.num_classes = len(pd.unique(self.mlcontext.train_y[self.mlcontext.train_y.columns.values[0]]))
    layer_count = self.args["layer_count"]+2
    num_epochs = self.args["num_epochs"]
    
    train_X = torch.tensor(X.astype('float32').values)
    min, _ = train_X.min(dim=0)
    max, _ = train_X.max(dim=0)       
    normalized_X = (train_X - min) / (max-min + 1)
    normalized_X = normalized_X
    
    self.le = LabelEncoder()
    y = self.le.fit_transform(y.values.ravel())
    train_y = torch.tensor(y)
    
    one_hot_encoded = torch.nn.functional.one_hot(train_y.to(torch.int64), self.num_classes).float()
    torch.manual_seed(random_seed)
    self.net = SimpleNetSoftmax(input_size, self.num_classes, layer_count)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(self.net.parameters(), lr=0.005)

    #Training loop
    for _ in range(num_epochs):
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
