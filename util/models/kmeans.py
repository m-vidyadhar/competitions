import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


__author__ = "Vidyadhar Mudium"


def plt_clusters(x_data):
    for cluster in x_data.cluster.unique():
        idx = x_data.cluster == cluster
        plt.scatter(x_data.loc[idx].x0, x_data.loc[idx].x1)
    plt.show()
    return


class KMeans():
    def __init__(self, x_data):
        self.x_data = x_data
        self.n_samples = len(x_data)
        self.hist_c = []
        pass
    
    
    def init_centres(self, n_clusters, n_attr):
        self.c_centres = np.random.randn(n_clusters, n_attr)
        self.hist_c.append(self.c_centres)
        pass
    
    
    def calc_distance(self, x_data, distance="euclidean"):
        return cdist(x_data.values, self.c_centres, metric=distance)
    

    def update_centers(self):
        self.c_centres = self.x_data.groupby("cluster").mean().values
        self.hist_c.append(self.c_centres)
        pass
    
    
    def update_clusters(self, x_data):
        return x_data.assign(cluster=self.calc_distance(x_data).argmin(axis=1))
    
    
    def fit(self, n_clusters, max_iter=1000, th=1e-5, verbose=250, plots=False):
        iteration = 0
        self.init_centres(n_clusters, self.x_data.shape[1])
        
        while (iteration <= max_iter):
            self.x_data = self.update_clusters(self.x_data.drop("cluster", axis=1, errors="ignore"))
            self.update_centers()
            
            if np.sum((self.hist_c[-1] - self.hist_c[-2]) ** 2) < th:
                print("Converged after {} iterations!\n".format(iteration))
                if plots:
                    plt_clusters(self.x_data)
                print(self.c_centres)
                break
            
            if (iteration % verbose == 0):
                print(self.c_centres)
                if plots:
                    plt_clusters(self.x_data)
            iteration += 1
        pass
    
    
    def predict(self, x_test):
        return self.update_clusters(x_test).cluster