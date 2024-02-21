from NextVisionML import MLContext
from NextVisionML import VarianceFilter, DecisionTreeClfr, OneHotClfr

mlcontext = MLContext()
#add feature selection methods
mlcontext.hooks.append(VarianceFilter(mlcontext)) 
#Add data prepocessing methods
mlcontext.train_methods.append(DecisionTreeClfr(mlcontext))
mlcontext.train_methods.append(OneHotClfr(mlcontext))
#select a model type

#start the training process
mlcontext.start_train_process()