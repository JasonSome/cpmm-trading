import numpy as np

class EWMA_Vol:
    def __init__(self, initial_vol, lambda_=0.97):
        self.lambda_ = lambda_
        self.sigma2 = initial_vol**2  # initial variance
    
    def update(self, r_t):
        self.sigma2 = self.lambda_ * self.sigma2 + (1 - self.lambda_) * r_t**2
        return np.sqrt(self.sigma2)