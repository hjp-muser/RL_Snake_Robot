from pca import PCA
from em_algorithm import EM
import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt


def write_csv(path=None):
    x = np.loadtxt("data/ynoise_data.txt")
    noise = x[:, 4:]
    prior = x[:, :4]
    # std = np.std(noise, axis=1)
    # keep_id = np.where(std < 0.16)[0]
    bias = noise - np.mean(noise, axis=1, keepdims=True)
    norm = np.sum(bias*bias, axis=1)
    acf = np.sum(bias[:, 1:]*bias[:, :-1], axis=1) / norm
    keep_id = np.where(acf > 0.95)[0]
    noise = noise[keep_id]
    prior = prior[keep_id]

    pca = PCA(noise, 15)
    pca_noise = pca.reduce_dimension()
    # EM algorithm
    clusters = 16
    a = EM(pca_noise, clusters)
    cluster_data, cluster_did, cluster_resp = a.clustering()
    # for cid in range(clusters):
    #     plt.figure("第{}类噪声".format(cid))
    #     for did in cluster_did[cid]:
    #         plt.plot(noise[did], color='r', alpha=a.responsibilities[cid][did])
    # plt.show()

    for i in range(clusters):
        print("第{}类数据量：".format(i), len(cluster_data[i]))
    csv_dat = defaultdict(list)

    def gray_code(n):
        if n == 1:
            return ['0', '1']
        return ['0' + i for i in gray_code(n - 1)] + ['1' + i for i in gray_code(n - 1)[::-1]]
    gc = gray_code(np.sqrt(clusters))

    for i, cid in enumerate(a.assignments):
        csv_dat['did'].append(keep_id[i])
        csv_dat['cid'].append(cid)
        csv_dat['prior'].append(np.array2string(prior[i], separator=',', floatmode='fixed'))
        csv_dat['noise'].append(np.array2string(noise[i], separator=',', floatmode='fixed'))
        csv_dat['gray'].append(gc[cid])

    mypd = pd.DataFrame(csv_dat)
    mypd.to_csv(path, encoding="utf-8-sig", mode="a", index=False)


def read_csv(path=None, rows=None, colums=None) -> pd.DataFrame:
    csv_dat = pd.read_csv(path, dtype={'did': int, 'cid': int, 'prior': str, 'noise': str, 'gray': str})
    if rows is not None:
        csv_dat = csv_dat.iloc[rows]
    if colums is not None:
        if type(colums) is not list:
            if type(colums) is int:
                csv_dat = csv_dat.iloc[:, colums]
            else:
                csv_dat = csv_dat.loc[:, colums]
        elif all(isinstance(x, int) for x in colums):
            csv_dat = csv_dat.iloc[:, colums]
        elif all(isinstance(x, str) for x in colums):
            csv_dat = csv_dat.loc[:, colums]

    return csv_dat


def get_dataset_size(path=None):
    csv_dat = pd.read_csv(path)
    return csv_dat.shape[0]


if __name__ == "__main__":
    dataset_path = 'data/1.csv'
    write_csv('data/1.csv')
    csv_df1 = read_csv(dataset_path)
    # csv_matrix1 = csv_df1.loc[:, ['cid', 'prior', 'gray']]
    # print(csv_matrix1.values)
    # print()
    # csv_df2 = read_csv(dataset_path, colums='prior')
    # print(csv_df2.values)
    #
    dataset_size = get_dataset_size(dataset_path)
    print("dataset_size = ", dataset_size)
    # csv_matrix = np.array([json.loads(lstr) for lstr in csv_df.values[:, 2]])
    # # print(csv_matrix)