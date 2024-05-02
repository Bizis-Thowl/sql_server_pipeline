
class ContinuousStatistics:
    def __init__(self, values):
        self.count = float(values.count())
        self.mean = float(values.mean())
        self.std = float(values.std())
        self.median = float(values.median())
        self.min = float(values.min())
        self.max = float(values.max())