from NextVisionML.train.interfaces.VarianceFilter import VarianceFilter
mlcontext = MLContext()
mlcontext.hooks.append(VarianceFilter(mlcontext)) 

mlcontext.start_train_process()