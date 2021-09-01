import numpy as np
import pandas as pd


class SampleData():
    def __init__(self, n_samples, n_attr, classes=None, clusters=None, class_balance=None) -> None:
        self.n_samples = n_samples
        self.classes = classes
        self.clusters = clusters
        self.class_balance = class_balance
        self.n_attr = n_attr
        pass

    def linear_reg_data(self, x_mean=10, const_term=True):
        col_names = ["x"+str(i) for i in range(1, self.n_attr + 1)]

        x_data = np.random.rand(self.n_samples, self.n_attr) * x_mean
        x_data = pd.DataFrame(x_data, columns=col_names)

        if const_term:
            x_data = x_data.assign(x0=1)

        self.true_beta = np.random.rand(self.n_attr + 1) * 3
        y = pd.Series(np.random.rand(self.n_samples)) + x_data.mul(self.true_beta).sum(axis=1)
        return (x_data, y)

    # def clustering_data(self):
    #     return (x_data, clusters)

    # def classification_data(self):
    #     return (x_data, labels)