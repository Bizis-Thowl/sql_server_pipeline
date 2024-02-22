import torch
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pandas as pd
from ...TrainInterface import TrainInterface
from ....util import update_object_attributes, create_objects
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from .temp_util import get_trust_scores

class OneHotClfr(TrainInterface):
    def __init__(self, mlContext):
        super().__init__(mlContext)
        
    def calculate(self, i, args):
        train_X = torch.tensor(self.mlContext.iter_train_X[i].astype('float32').values)
        min, _ = train_X.min(dim=0)
        max, _ = train_X.max(dim=0)       
        normalized_X = (train_X - min) / (max-min + 1)
        normalized_X = normalized_X
        train_y = torch.tensor(self.mlContext.iter_objs[i]["integer_encoded_train_y"])
        
        random_seed = args["random_seed"]
        input_size = len(self.mlContext.iter_train_X[i].columns)
        num_classes = 5
        layer_count = args["layer_count"]+3
        num_epochs = args["num_epochs"]*10
        one_hot_encoded = torch.nn.functional.one_hot(train_y.to(torch.int64), num_classes).float()
        torch.manual_seed(random_seed)
        net = SimpleNetSoftmax(input_size, num_classes, layer_count)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.005)

        #Training loop
        for _ in range(num_epochs):
            outputs = net(normalized_X)
            loss = criterion(outputs, one_hot_encoded)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #Evluation     
        test_X = torch.tensor(self.mlContext.iter_test_X[i].astype('float32').values)
        min, _ = test_X.min(dim=0)
        max, _ = test_X.max(dim=0)
        test_X_n = (test_X - min) / (max-min + 1)

        test_X_n = test_X_n
        pred = net(test_X)
        _ , pred = torch.max(pred, 1)
        balanced_accuracy = balanced_accuracy_score(self.mlContext.iter_objs[i]["integer_encoded_test_y"], pred)   
        
        #Cache for upload
        self.mlContext.iter_objs[i]["one_hot_pred"] = pred.numpy()
        self.mlContext.iter_objs[i]["one_hot_balanced_accuracy"] = balanced_accuracy        
        self.mlContext.iter_objs[i]["one_hot_trust_scores"] = get_trust_scores(self.mlContext.iter_objs[i]["integer_encoded_test_y"], pred)
        
        return balanced_accuracy
    
    def upload(self, i):      
        self.mlContext.iter_objs[i]["model"]["one_hot"] = update_object_attributes(self.mlContext.context, self.mlContext.iter_objs[i]["model"]["one_hot"],
                                                                        path_to_model = "one_hot")  
        self.mlContext.iter_objs[i]["model_score_one_hot"] = update_object_attributes(self.mlContext.context, self.mlContext.iter_objs[i]["model_score_one_hot"],
                             balanced_accuracy_score = self.mlContext.iter_objs[i]["one_hot_balanced_accuracy"])
        pred_mem = self.mlContext.iter_objs[i]["one_hot_pred"]
        label_encoder = self.mlContext.iter_objs[i]["label_encoder_one_hot"]
        
        pred_labeled = label_encoder.inverse_transform(pred_mem)
        df = pd.DataFrame()                
        df["pred"] = pred_labeled
        df["trust_score"] = self.mlContext.iter_objs[i]["one_hot_trust_scores"] 
        df["datapoint_id"] = self.mlContext.test_db_indexes
        df["model_id"] = self.mlContext.iter_objs[i]["model"]["one_hot"].id
        df.reset_index(inplace=True)
        df.to_sql("label_categorical", con = self.mlContext.engine, index = False, if_exists='append')
        
    def populate(self, i):
        args = {
            "random_seed": hp.randint("random_seed", 1000),
            "layer_count": hp.randint("layer_count", 3), 
            "num_epochs": hp.randint("num_epochs", 15),
        }       
        self.mlContext.iter_args[i].update(args)  
        self.mlContext.iter_objs[i]["model"]["one_hot"] = update_object_attributes(self.mlContext.context, self.mlContext.iter_objs[i]["model"]["one_hot"],
                                                                        path_to_model = "one_hot") #TODO: Refactor
        label_encoder = LabelEncoder()
        integer_encoded_train = label_encoder.fit_transform(self.mlContext.iter_train_y[i].values.ravel())
        self.mlContext.iter_objs[i]["integer_encoded_train_y"] = integer_encoded_train
        integer_encoded_test = label_encoder.transform(self.mlContext.iter_test_y[i].values.ravel())
        self.mlContext.iter_objs[i]["integer_encoded_test_y"] = integer_encoded_test
        self.mlContext.iter_objs[i]["label_encoder_one_hot"] = label_encoder
                 
class SimpleNetSoftmax(torch.nn.Module):
    def __init__(self, input_size, output_size, layer_count):
        super(SimpleNetSoftmax, self).__init__()
        self.call_super_init
        self.fc = torch.nn.ModuleList()
        self.lc = layer_count
        for _ in range(layer_count-1):
            self.fc.append(torch.nn.Linear(input_size, input_size*2))
            input_size*=2
        self.fc.append(torch.nn.Linear(input_size, output_size))
        #self.sigmoid = torch.sigmoid

    def forward(self, X):
        y = X
        for layer in self.fc:
            y = layer(y)
        y = torch.softmax(y, dim=1)
        return y
