import numpy as np

class bandit:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
    
    def hit(self):
        return np.random.normal(self.mean, self.var)