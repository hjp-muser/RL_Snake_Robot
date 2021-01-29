import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DimensionValueError(ValueError):
    """定义异常类"""
    pass


class PCA(object):
    """定义PCA类"""

    def __init__(self, X: np.ndarray, n_components=None):
        self.X = X
        self.dimension = X.shape[1]

        if n_components and n_components >= self.dimension:
            raise DimensionValueError("n_components error")

        self.n_components = n_components

    def cov(self):
        """求x的协方差矩阵"""
        x_T = np.transpose(self.X)  # 矩阵转秩
        x_cov = np.cov(x_T)  # 协方差矩阵
        return x_cov

    def get_feature(self):
        """求协方差矩阵C的特征值和特征向量"""
        x_cov = self.cov()
        fval, fvec = np.linalg.eig(x_cov)
        m = fval.shape[0]
        fv = np.hstack((fval.reshape((m, 1)), fvec))
        fv_sort = fv[np.argsort(-fv[:, 0])]
        return fv_sort

    def explained_varience_(self):
        fv_sort = self.get_feature()
        return fv_sort[:, 0]

    def paint_varience_(self):
        explained_variance_ = self.explained_varience_()
        plt.figure()
        plt.plot(explained_variance_, 'k')
        plt.xlabel('n_components', fontsize=16)
        plt.ylabel('explained_variance_', fontsize=16)
        plt.show()

    def reduce_dimension(self):
        """指定维度降维和根据方差贡献率自动降维"""
        fv_sort = self.get_feature()
        varience = self.explained_varience_()

        if self.n_components:  # 指定降维维度
            p = fv_sort[0:self.n_components, 1:]
            res = np.dot(p, np.transpose(self.X))  # 矩阵叉乘

        else:
            varience_sum = sum(varience)  # 利用方差贡献度自动选择降维维度
            varience_radio = varience / varience_sum

            varience_contribution = 0
            for R in range(self.dimension):
                varience_contribution += varience_radio[R]  # 前R个方差贡献度之和
                if varience_contribution >= 0.9:
                    break

            p = fv_sort[0:R + 1, 1:]  # 取前R个特征向量
            res = np.dot(p, np.transpose(self.X))  # 矩阵叉乘
            # self.paint_varience_()
        return np.transpose(res)


if __name__ == '__main__':
    from em_algorithm import EM
    x = np.loadtxt("data/ynoise_data.txt")
    x = x[:, 4:]

    bias = x - np.mean(x, axis=1, keepdims=True)
    norm = np.sum(bias*bias, axis=1)
    acf = np.sum(bias[:, 1:]*bias[:, :-1], axis=1) / norm
    row = np.where(acf > 0.95)

    # std = np.std(x, axis=1)
    # row = np.where(std < 0.16)
    x = x[row]

    pca = PCA(x, 15)
    y = pca.reduce_dimension()

    plt.figure("噪声图像")
    for data in x:
        plt.plot(data, color='r')

    #    fig = plt.figure()
    #    ax = fig.add_subplot(111, projection='3d')
    #    plt.scatter(y[:, 0], y[:, 1], edgecolors="black")
    #    ax.scatter(y[:,0], y[:,1], y[:,2])

    # EM algorithm
    clusters = 16
    a = EM(y, clusters)
    cluster_data, cluster_did, cluster_resp = a.clustering()
    for cid in range(clusters):
        plt.figure("第{}类噪声".format(cid))
        # for did in cluster_did[cid]:
        #     plt.plot(x[did], color='r', alpha=a.responsibilities[cid][did])
        plt.plot(x[cluster_did[cid][15]], color='r', alpha=a.responsibilities[cid][cluster_did[cid][15]])
    plt.show()

    for i in range(clusters):
        print("第{}类数据量：".format(i), len(cluster_data[i]))
