from notebooks.NextVisionML.train.XAIInterface import XAIInterface
import pandas as pd
import dice_ml
from dice_ml.utils import helpers # helper functions
from sklearn.model_selection import train_test_split

class CounterFactuals(XAIInterface):
    def __init__(self, mlContext):
        super().__init__(mlContext)
    
    def call(self, i, args):
        train_dataset = pd.Concat([self.mlContext.iter_train_X[i], self.mlContext.iter_train_y[i]], axis = 0)
        d = dice_ml.Data(
            dataframe=train_dataset
            continuous_features=self.mlContext.iter_train_X[i].columns,
            outcome_name="Risk-Level")

        # Using sklearn backend
        m = dice_ml.Model(model=self.mlContext.iter_objs[i]["model"], backend="sklearn")
        # Using method=random for generating CFs
        exp = dice_ml.Dice(d, m, method="random")