from NextVisionML import MLContext
from NextVisionML import VarianceFilter, DecisionTreeClfr, OneHotClfr, CorrelationFilter, MutualInfoFilter

mlcontext = MLContext()
#add feature selection methods
mlcontext.train_preparation_methods.append(VarianceFilter(mlcontext)) 
mlcontext.train_preparation_methods.append(CorrelationFilter(mlcontext)) 
mlcontext.train_preparation_methods.append(MutualInfoFilter(mlcontext)) 
#Add data prepocessing methods
mlcontext.train_methods.append(OneHotClfr(mlcontext))
mlcontext.train_methods.append(DecisionTreeClfr(mlcontext))
#select a model type
#start the training process
mlcontext.start_train_process()