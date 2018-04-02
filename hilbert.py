"""
Occupancy Mapping
Ransalu Senanayake
"""

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import SGDClassifier

class HilbertMap():
    def __init__(self, gamma=0.075*0.814, grid=None, cell_resolution=(5, 5), cell_max_min=None, X=None, alpha=0.001):
        """
        :param gamma: RBF bandwidth
        :param grid: if there are prespecified locations to hinge the RBF
        :param cell_resolution: if 'grid' is 'None', resolution to hinge RBFs
        :param cell_max_min: if 'grid' is 'None', realm of the RBF field
        :param X: a sample of lidar locations to use when both 'grid' and 'cell_max_min' are 'None'
        """
        self.gamma = gamma
        if grid is not None:
            self.grid = grid
        else:
            self.grid, self.xx, self.yy = self.__calc_grid_auto(cell_resolution, cell_max_min, X)
        self.intercept_, self.coef_, self.sigma_ = [0], [0], [0]
        self.scan_no = 0
        #self.classifier = SGDClassifier(loss="log", penalty="elasticnet", alpha=alpha)
        self.classifier = SGDClassifier(loss="log", penalty="l2", alpha=alpha, max_iter=1000, fit_intercept=False) #fit intecept artificially added in __sparse_features

    def __calc_grid_auto(self, cell_resolution, max_min, X):
        """
        :param X: a sample of lidar locations
        :param cell_resolution: resolution to hinge RBFs as (x_resolution, y_resolution)
        :param max_min: realm of the RBF field as (x_min, x_max, y_min, y_max)
        :return: numpy array of size (# of RNFs, 2) with grid locations
        """

        if max_min is None:
            # if 'max_min' is not given, make a boundarary based on X
            # assume 'X' contains samples from the entire area
            expansion_coef = 1.2
            x_min, x_max = expansion_coef*X[:, 0].min(), expansion_coef*X[:, 0].max()
            y_min, y_max = expansion_coef*X[:, 1].min(), expansion_coef*X[:, 1].max()
        else:
            x_min, x_max = max_min[0], max_min[1]
            y_min, y_max = max_min[2], max_min[3]

        xx, yy = np.meshgrid(np.arange(x_min, x_max, cell_resolution[0]), \
                             np.arange(y_min, y_max, cell_resolution[1]))
        grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))

        return grid, xx, yy

    def __sparse_features(self, X):
        """
        :param X: inputs of size (N,2)
        :return: hinged features with intercept of size (N, # of features + 1)
        """
        rbf_features = rbf_kernel(X, self.grid, gamma=self.gamma)
        #print(rbf_features)
        return np.hstack((np.ones(X.shape[0])[:, np.newaxis], rbf_features))

    def fit(self, X, y):
        kernelzed_X = self.__sparse_features(X)
        self.classifier.fit(kernelzed_X, y)

    def predict_prob(self, X_q):
        kernelzed_X_q = self.__sparse_features(X_q)
        return self.classifier.predict_proba(kernelzed_X_q)[:,1]

