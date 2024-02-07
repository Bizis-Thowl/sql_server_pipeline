class XAIInterface:
    def __init__(self, mlContext):
        self.mlContext = mlContext
        
    def call(self, i, args): # Derive Hyperopt args from method not from context.iter_args[i]!!!!!
        pass