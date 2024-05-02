
class CategoricalStatistics:
    def __init__(self, values):
        self.count = int(values.count())
        original_dict = values.value_counts().to_dict()
        self.value_counts = {str(key): value for key, value in original_dict.items()}