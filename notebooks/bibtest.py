from NextVisionML import MLContext
from NextVisionML import VarianceFilter, DecisionTreeClfr, PcaUnsupervised

mlcontext = MLContext()
#add feature selection methods
mlcontext.train_preparation_methods.append(VarianceFilter(mlcontext)) 
#Add data prepocessing methods
mlcontext.train_methods.append(DecisionTreeClfr(mlcontext))
mlcontext.train_methods.append(PcaUnsupervised(mlcontext))
#select a model type

#start the training process
mlcontext.start_train_process()