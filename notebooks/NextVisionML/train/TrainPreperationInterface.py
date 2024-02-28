class TrainPreperationInterface:
    def __init__(self, mlContext):
        self.mlContext = mlContext
        
    def claculate(self, i, args): # Derive Hyperopt args from method not from context.iter_args[i]!!!!!
        pass
    
    def populate(self, i):
        pass