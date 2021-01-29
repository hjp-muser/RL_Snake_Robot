from collections import defaultdict
import numpy as np


class KMeans:
    def __init__(self, X: np.ndarray, K: int):
        """
        K-Means algorithm

        Parameters
        ----------
        X : datasets
        K : cluster number
        """
        self.X = np.array(X)
        self.K = K
        self.data_size = X.shape[0]
        self.data_dimension = X.shape[1]
        self.has_clustered = False
        self.assignments = None

    def _fit(self, max_iter=None):
        """
        K-Means algorithm process:
        1. Generate K centers
        2. Assign data to K clusters
        3. update centers
        """
        if self.has_clustered is False:
            init_centers = self._generate_centers()
            assignments = self._assign_points(init_centers)
            old_assignments = None
            iters = 0
            while assignments != old_assignments and (max_iter is None or iters < max_iter):
                new_centers = self._update_centers(assignments)
                old_assignments = assignments
                assignments = self._assign_points(new_centers)
                iters += 1
            # detect_res = self._detect_empty_cluster(assignments)

            self.has_clustered = True
            self.assignments = assignments

    def _generate_centers(self, method="maximum"):
        if method == "sample":
            rand = np.random.choice(self.data_size, self.K, replace=False)
            cluster_centers = self.X[rand, :]
        elif method == "random":
            cluster_centers = np.random.uniform(low=self.X.min(axis=0),
                                                high=self.X.max(axis=0),
                                                size=(self.K, self.data_dimension))
        elif method == "maximum":
            cluster_centers = []
            first_center = self.X[0]
            cluster_centers.append(first_center)
            while len(cluster_centers) < self.K:
                max_dis = 0
                max_i = 0
                for i in range(self.data_size):
                    dis = np.inf
                    for center in cluster_centers:
                        dis = min(np.sqrt(np.sum((self.X[i]-center)**2)), dis)
                    if dis > max_dis:
                        max_dis = dis
                        max_i = i
                cluster_centers.append(self.X[max_i])
            cluster_centers = np.array(cluster_centers)
        else:
            raise NotImplementedError
        return cluster_centers

    def _assign_points(self, centers):
        assignments = []
        for point in self.X:
            shortest = np.inf  # positive infinity
            shortest_index = 0
            for i in range(len(centers)):
                val = np.sqrt(np.sum((point - centers[i]) ** 2))
                if val < shortest:
                    shortest = val
                    shortest_index = i
            assignments.append(shortest_index)
        return assignments

    def _update_centers(self, assignments):
        new_cluster = defaultdict(list)
        new_centers = []
        for assignment, point in zip(assignments, self.X):
            new_cluster[assignment].append(point)

        for x in new_cluster.values():
            new_centers.append(np.mean(x, axis=0))

        return new_centers

    def clustering(self, max_iter=None):
        self._fit(max_iter)
        cluster = defaultdict(list)
        for assignment, point in zip(self.assignments, self.X):
            cluster[assignment].append(point)
        return cluster

    def get_cluster_centers(self):
        cluster = self.clustering()
        cluster_centers = []
        for x in cluster.values():
            cluster_centers.append(np.mean(x, axis=0))
        return np.array(cluster_centers)

    def get_cluster_variances(self):
        cluster = self.clustering()
        cluster_variance = []
        for x in cluster.values():
            cluster_variance.append(np.std(x, axis=0))
        return np.array(cluster_variance)

    def get_cluster_proportions(self):
        cluster = self.clustering()
        cluster_proportion = []
        for x in cluster.values():
            cluster_proportion.append(len(x) / self.data_size)
        return np.array(cluster_proportion)

    def get_assignments(self):
        self._fit()
        return self.assignments


if __name__ == '__main__':
    points = [
        [1, 2],
        [2, 1],
        [3, 1],
        [5, 4],
        [5, 5],
        [6, 5],
        [10, 8],
        [7, 9],
        [11, 5],
        [14, 9],
        [14, 14],
        [50, 50],
        [50, 51],
        [99, 199],
        [99, 188]
    ]
    a = KMeans(np.array(points), 3)
    clusters = a.clustering()
    print(a.get_cluster_centers())
    print(a.get_cluster_variances())
    print(a.get_cluster_proportions())
