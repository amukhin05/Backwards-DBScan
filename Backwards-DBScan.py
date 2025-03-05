import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from scipy.spatial.distance import cdist

X, _ = make_blobs(n_samples=1000, centers=3, cluster_std=[1.3, 3.0, 0.6])
plt.scatter(X[:,0],X[:,1],s=10, alpha=0.8)
plt.show()

class DBC():

    def __init__(self, X, min_samples, eps):
        self.X = X
        self.min_samples = min_samples
        self.eps = eps

    def dbscanIT(self):
        remaining_points = self.X.copy()
        all_labels = np.full(len(self.X), -1)
        indices = np.arange(len(self.X))

        while len(remaining_points) > 0 and self.eps < 3.0:
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = db.fit_predict(remaining_points)

            current_labels = labels
            non_noise_mask = current_labels != -1
            if np.any(non_noise_mask):
                all_labels[indices[non_noise_mask]] = current_labels[non_noise_mask]

            remaining_points = remaining_points[current_labels == -1]
            indices = indices[current_labels == -1]

            self.eps += 0.4

        return all_labels

    def dbscanBack(self):
        remaining_points = self.X.copy()
        all_labels = np.full(len(self.X), -1)
        indices = np.arange(len(self.X))

        prev_all_labels = np.copy(all_labels)

        while len(remaining_points) > 0 and self.eps > 0.2:
            if (self.eps < 1):
                self.min_samples /= 1.6
            if (self.eps < 0.5):
                self.min_samples /= 1.3
            temp_min = self.min_samples
            self.min_samples = int(self.min_samples)

            db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = db.fit_predict(remaining_points)

            current_labels = labels
            non_noise_mask = current_labels != -1

            if np.any(non_noise_mask):
                all_labels[indices[non_noise_mask]] = current_labels[non_noise_mask]

            for idx, label in enumerate(current_labels):
                if label != -1:
                    if all_labels[indices[idx]] != -1 and all_labels[indices[idx]] != label:
                        if self.eps < self.prev_eps:
                            all_labels[indices[idx]] = label
                    else:
                        all_labels[indices[idx]] = label

            for idx, label in enumerate(current_labels):
                if label != -1 and all_labels[indices[idx]] != label:
                    prev_label = prev_all_labels[indices[idx]]
                    if prev_label != -1 and prev_label == all_labels[indices[idx]]:
                        all_labels[indices[idx]] = prev_label

            remaining_points = remaining_points[non_noise_mask]
            indices = indices[non_noise_mask]

            self.prev_eps = self.eps
            self.min_samples = temp_min
            self.eps /= 1.4

            prev_all_labels = np.copy(all_labels)

        unique_labels = np.unique(all_labels)
        for label in unique_labels:
            if label == -1:
                continue

            cluster_points_indices = np.where(all_labels == label)[0]
            cluster_points = self.X[cluster_points_indices]

            if len(cluster_points) <= 10:
                other_cluster_labels = unique_labels[unique_labels != label]
                other_cluster_points = self.X[np.isin(all_labels, other_cluster_labels)]
                distances = cdist(cluster_points, other_cluster_points)

                for idx, point in enumerate(cluster_points):
                    nearest_point_idx = np.argmin(distances[idx])
                    nearest_cluster_idx = np.where(np.isin(self.X, other_cluster_points[nearest_point_idx]))[0][0]
                    nearest_cluster_label = all_labels[nearest_cluster_idx]
                    all_labels[cluster_points_indices[idx]] = nearest_cluster_label

        return all_labels

#clustering = DBC(X, 7, 1.4).dbscanIT()
clustering = DBC(X, 30, 2).dbscanBack()
if clustering.size == 0:
    print("No clusters found.")
else:
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    color_map = np.array([colors[label % len(colors)] if label != -1 else 'gray' for label in clustering])

    plt.scatter(X[:, 0], X[:, 1], color=color_map, s=10, alpha=0.8)
    plt.title("Iterative DBSCAN Clustering")
    plt.show()
