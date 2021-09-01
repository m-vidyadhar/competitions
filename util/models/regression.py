import numpy as np
import pandas as pd


__author__ = "Vidyadhar Mudium"


class LinearRegression():
    def __init__(self, x_data, y, add_const=True):
        self.x_data = x_data
        self.y = y
        self.n_samples = len(y)
        self.n_attr = x_data.shape[1] + int(not add_const)
        self.add_const = add_const
        pass
    
    def add_constant(self, x_data):
        return x_data.assign(x0=1)
    
    def init_weights(self):
        self.beta = np.random.rand(self.n_attr + 1)
        pass
    
    def mse_loss(self, n_digits=4):
        return round((self.y - self.y_pred).pow(2).sum() / self.n_samples, n_digits)
    
    def weight_gradient(self):
        return - self.x_data.mul((self.y - self.y_pred) * 2, axis=0).sum() / self.n_samples
    
    def update_weights(self, eta):
        self.beta = self.beta - eta * self.weight_gradient().values
    
    def fit(self, eta, epochs=250, th=1e-3, verbose=20, decay_f=2, lr_update=100):
        if self.add_const:
            self.x_data = self.add_constant(self.x_data)
        
        self.init_weights()
        
        for epoch in range(epochs):
            self.y_pred = self.x_data.mul(self.beta).sum(axis=1)
            if (epoch % verbose == 0):
                print("Epoch: {}; MSE: {}".format(epoch, self.mse_loss()))
            
            if (epoch % lr_update == 0):
                eta /= decay_f
                print("Learning updated to {} at Epoch: {}\n".format(round(eta, 3), epoch))
            self.update_weights(eta)
            
            if self.mse_loss() < th:
                print("Optimal Solution reached at {} with MSE={}!".format(epoch, self.mse_loss()))
                break
        pass
    
    def predict(self, x_test, add_const=True):
        if add_const:
            x_test = self.add_const(x_test)
        
        return x_test.mul(self.beta).sum(axis=1)