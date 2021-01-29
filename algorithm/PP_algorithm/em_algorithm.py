import numpy as np
from scipy.stats import multivariate_normal
from numpy import pi, sin, cos
from collections import defaultdict
import matplotlib.pyplot as plt
from kmeans import KMeans


class EM:
    def __init__(self, X, K=2, init_cov_size=120):
        self.X = X
        self.K = K
        self.datapoints = self.X.shape[0]
        self.dims = self.X.shape[1]
        self.it = 0
        self.init_cov_size = init_cov_size

        # initialize with random points and identitiy matrices
        # self.cluster_centers = np.random.uniform(low=self.X.min(axis=0),
        #                                          high=self.X.max(axis=0),
        #                                          size=(self.K, self.X.shape[1]))

        # init means with random points from the data; seems to result in fewer singulartities
        # rand = np.random.choice(self.datapoints, self.K, replace=False)
        # self.cluster_centers = self.X[rand, :]
        # self.cluster_covs = np.stack([np.eye(self.dims) * self.init_cov_size] * self.K, axis=0)
        # self.mixing_coeffs = np.full(self.K, 1 / self.K)

        # init with kmeans
        self.cluster_centers = None
        self.cluster_covs = None
        self.mixing_coeffs = None
        self._init_with_kmeans()
        self.responsibilities = None
        self.assignments = None

    def _init_with_kmeans(self):
        kmeans = KMeans(self.X, self.K)
        self.cluster_centers = kmeans.get_cluster_centers()
        cluster_variances = kmeans.get_cluster_variances()
        self.cluster_covs = np.array([np.eye(self.dims) * v for v in cluster_variances])
        self.mixing_coeffs = kmeans.get_cluster_proportions()

    def _fit(self, max_iter=10):
        for i in range(max_iter):
            # Expectation
            self.responsibilities = self._expectation()
            # Maximization
            self._maximization()
            self.it += 1
            print("iteration: ", i, "mixing_coeffs: ", self.mixing_coeffs[:5])
        self._resort_clusterid()
        # self._unsort_clusterid()

    def _expectation(self):
        tripel = zip(self.cluster_centers, self.cluster_covs, self.mixing_coeffs)
        responsibilities = np.zeros((self.K, self.datapoints))
        divisor_sum = np.zeros(self.datapoints)

        for i, (mean, cov, mixing_coeff) in enumerate(tripel):
            resp_numerator = mixing_coeff * multivariate_normal.pdf(self.X, mean, cov, allow_singular=True)
            responsibilities[i] = resp_numerator
            divisor_sum += resp_numerator
        responsibilities /= divisor_sum
        return responsibilities

    def _maximization(self):
        for i, resp in enumerate(self.responsibilities):
            Nk = resp.sum()
            if Nk <= self.datapoints / (self.K * 10):
                # catch near singularities
                print("Singularity detected. Resetting mu and cov.")

                # choosing new mean uniformly random
                # new_mean = np.random.uniform(low=self.X.min(axis=0),
                #                              high=self.X.max(axis=0))

                # choosing random points form X as mean
                rand = np.random.choice(self.datapoints, replace=False)
                new_mean = self.X[rand, :]

                new_cov = np.eye(self.dims) * self.init_cov_size
            else:
                new_mean = 1 / Nk * (resp[:, np.newaxis] * self.X).sum(axis=0)
                unweighted_product = np.einsum('ji,jk->jik', (self.X - new_mean), (self.X - new_mean))
                cov_sum = (resp[:, np.newaxis, np.newaxis] * unweighted_product).sum(axis=0)
                new_cov = 1 / Nk * cov_sum
            new_mixing_coeff = Nk / self.datapoints

            self.cluster_centers[i] = new_mean
            self.cluster_covs[i] = new_cov
            self.mixing_coeffs[i] = new_mixing_coeff

    def _resort_clusterid(self):
        sorted_tuples = sorted(enumerate(self.cluster_centers), key=lambda id_centers: np.linalg.norm(id_centers[1]))
        # sorted_tuples = sorted(enumerate(self.cluster_centers), key=lambda id_centers: id_centers[1][0])
        sorted_subscrs = [t[0] for t in sorted_tuples]
        self.responsibilities = self.responsibilities[sorted_subscrs]
        sorted_clusterid = self.responsibilities.argmax(axis=0)
        self.assignments = sorted_clusterid
        self.cluster_centers = self.cluster_centers[sorted_subscrs]
        self.cluster_covs = self.cluster_covs[sorted_subscrs]
        self.mixing_coeffs = self.mixing_coeffs[sorted_subscrs]

    def _unsort_clusterid(self):
        unsorted_clusterid = self.responsibilities.argmax(axis=0)
        self.assignments = unsorted_clusterid

    def clustering(self, max_iter=50):
        self._fit(max_iter)
        cluster_data = defaultdict(list)
        for cid, point in zip(self.assignments, self.X):
            cluster_data[cid].append(point)

        cluster_did = defaultdict(list)
        for did, cid in enumerate(self.assignments):
            cluster_did[cid].append(did)

        cluster_resp = defaultdict(list)
        for did, cid in enumerate(self.assignments):
            cluster_resp[cid].append(self.responsibilities[cid][did])
        return cluster_data, cluster_did, cluster_resp


def oval(cov, num_points=100, radius=1):
    arcs = np.linspace(0, 2 * pi, num_points)
    x = radius * sin(arcs)
    y = radius * cos(arcs)

    xy = np.array(list(zip(x, y)))
    x, y = zip(*xy.dot(cov))
    return x, y


def make_plot(a, X, dims):
    fig = plt.figure(figsize=(6, 5))
    plt.title("EM iteration {}".format(a.it))

    colors = ['g', 'r', 'c', 'm', 'y', 'b', 'k']

    if dims == "2d":
        # selcect elements based on expectation
        x, y = zip(*X)
        try:
            # plt.scatter(x, y, edgecolors="black",c=a.responsibilities[0],cmap='RdYlGn')
            color_arr = [colors[i] for i in a.responsibilities.argmax(axis=0)]
            plt.scatter(x, y, edgecolors="black", c=color_arr)
        except AttributeError:
            plt.scatter(x, y, edgecolors="black", color='y')
        for i in range(a.cluster_centers.shape[0]):
            # plot centers
            plt.scatter(a.cluster_centers[i, 0], a.cluster_centers[i, 1], s=250, color=colors[i], edgecolors="white")

            # plot ovals that show the shape of the  variances
    #        x, y = oval(a.cluster_covs[i],radius=2)
    #        x += a.cluster_centers[i,0]
    #        y += a.cluster_centers[i,1]
    #        plt.plot(x, y,linewidth=5,color=colors[i])
    elif dims == "3d":
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = zip(*X)
        try:
            #        plt.scatter(x, y, edgecolors="black",c=a.responsibilities[0],cmap='RdYlGn')
            color_arr = [colors[i] for i in a.responsibilities.argmax(axis=0)]
            ax.scatter(x, y, z, edgecolors="black", c=color_arr)
        except AttributeError:
            ax.scatter(x, y, z, edgecolors="black", color='y')
        for i in range(a.cluster_centers.shape[0]):
            # plot centers
            ax.scatter(a.cluster_centers[i, 0], a.cluster_centers[i, 1], a.cluster_centers[i, 2], s=250,
                       color=colors[i], edgecolors="white")
    else:
        pass


if __name__ == '__main__':
    # import dataset
    import pandas as pd

    X = pd.read_csv("data/2d-em.csv", header=None).values
    # plot dataset
    x, y = zip(*X)
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, edgecolors="black")
    a = EM(X, 2, init_cov_size=2)
    cluster_data, cluster_did, cluster_resp = a.clustering()
    # make_plot(a, X, "2d")
