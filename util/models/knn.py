import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from scipy.stats import mode


__author__ = "Vidyadhar Mudium"


def _get_neighbors(x1, x2, k, distance="euclidean") -> tuple:
    dist_df = cdist(x1.values, x2.values, metric=distance)
    return (dist_df, np.argpartition(dist_df, kth=k, axis=1)[:, :k])


def _get_weights(dist_df, neighbors):
    weights = np.take_along_axis(dist_df, neighbors, axis=1)
    return np.divide((weights ** -2), np.sum(weights ** -2, axis=1).reshape(-1, 1))


class kNNClassifer():
    def __init__(self, x_data, labels) -> None:
        self.x_data = x_data
        self.labels = labels
        pass


    def _weighted_count(self, ngh_labels, weights):
        minlength = np.max(ngh_labels) + 1
        return np.array([
            np.bincount(arr, weights=weights[i], minlength=minlength)
            for (i, arr) in enumerate(ngh_labels)
        ])


    def predict(self, x_test, k=3, weighted=True):
        self.dist_df, self.neighbors = _get_neighbors(x_test, self.x_data, k)
        self.ngh_labels = self.labels.values[self.neighbors]
        if not weighted:
            return np.ravel(mode(self.ngh_labels, axis=1)[0])

        self.weights = _get_weights(self.dist_df, self.neighbors)
        return np.argmax(self._weighted_count(self.ngh_labels, self.weights), axis=1)


class kNNRegressor():
    def __init__(self, x_data, y) -> None:
        self.x_data = x_data
        self.y = y
        pass


    def predict(self, x_test, k=3, weighted=True):
        self.dist_df, self.neighbors = _get_neighbors(x_test, self.x_data, k)
        self.ngh_y = self.y.values[self.neighbors]
        if not weighted:
            return np.mean(self.ngh_y, axis=1)
        
        self.weights = _get_weights(self.dist_df, self.neighbors)
        return np.sum(self.weights * self.ngh_y, axis=1)