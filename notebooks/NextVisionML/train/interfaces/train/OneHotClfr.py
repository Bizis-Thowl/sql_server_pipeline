import torch
import hyperopt as hp
import pandas as pd
from ...TrainInterface import TrainInterface
from ....util import update_object_attributes, create_objects
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score

class OneHotClfr(TrainInterface):
    def __init__(self, mlContext):
        super().__init__(mlContext)
        
    def calculate(self, i, args):
        #TODO:hyperparameter in sql
        train_X = torch.tensor(self.mlContext.iter_train_X[i].values)
        train_y = torch.tensor(self.mlContext.iter_objs[i]["integer_encoded_train_y"])
        random_seed = self.mlContext.iter_args[i]["random_seed"]
        input_size = len(train_X.columns)
        num_classes = 5  #TODO:not hadcoded
        layer_count = self.mlContext.iter_args[i]["layer_count"]
        num_epochs = self.mlContext.iter_args[i]["num_epochs"]
                      
        one_hot_encoded = torch.nn.functional.one_hot(train_y, num_classes)
        
        torch.manual_seed(random_seed)
        net = SimpleNetSigmoid(input_size, layer_count)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

        #Training loop
        for epoch in range(num_epochs):
            outputs = net(one_hot_encoded)
            loss = criterion(outputs, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #Evluation
        test_X = torch.tensor(self.mlContext.iter_test_X[i].values)        
        pred = net(test_X)
        pred = torch.max(pred, 1) #Reverse One Hot
        pred_mem = pred.numpy()
        balanced_accuracy = balanced_accuracy_score(self.mlContext.iter_objs[i]["integer_encoded_test_y"], pred_mem)       
        
        #Cache for upload
        self.mlContext.iter_objs[i]["one_hot_pred"] = pred_mem
        self.mlContext.iter_objs[i]["one_hot_balanced_accuracy"] = balanced_accuracy
              
        return balanced_accuracy   
    
    def upload(self, i):        
        update_object_attributes(self.mlContext.context, self.mlContext.iter_objs[i]["model_score"],
                             balanced_accuracy_score = self.mlContext.iter_objs[i]["balanced_accuracy"])
        pred_mem = self.mlContext.iter_objs[i]["one_hot_pred"]
        label_encoder = self.mlContext.iter_objs[i]["label_encoder_one_hot"]
        
        pred_labeled = label_encoder.inverse_transform(pred_mem)
        df = pd.DataFrame()                
        df["pred"] = pred_labeled
        df["datapoint_id"] = self.mlContext.test_db_indexes
        df["model_id"] = self.mlContext.iter_objs[i]["model"].id
        df.reset_index(inplace=True)
        df.to_sql("metmast", con = self.mlContext.engine, index = False, if_exists='append')
        
    def populate(self, i):
        args = {
            "random_seed": hp.randint("random_seed", 1000),
            "layer_count": hp.randint("layer_count", 3), #iterargs
            "num_epochs": hp.randint("num_epochs", 15), #iterargs
        }       
        self.mlContext.iter_args[i].update(args)  
        self.mlContext.iter_objs[i]["models"]["one_hot"] = update_object_attributes(self.mlContext, self.mlContext.iter_objs[i]["model"],
                                                                        path_to_model = "one_hot") #TODO: Refactor
        label_encoder = LabelEncoder()
        integer_encoded_train = label_encoder.fit_transform(self.mlContext.iter_train_y[i])
        self.mlContext.iter_objs[i]["integer_encoded_train_y"] = integer_encoded_train
        integer_encoded_test = label_encoder.transform(self.mlContext.iter_test_y[i])
        self.mlContext.iter_objs[i]["integer_encoded_test_y"] = integer_encoded_test
        self.mlContext.iter_objs[i]["label_encoder_one_hot"] = label_encoder
                 
class SimpleNetSigmoid(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNetSigmoid, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)
        self.sigmoid = torch.Sigmoid()

    def forward(self, X):
        y = self.fc(X)
        y = self.sigmoid(y)
        return y
