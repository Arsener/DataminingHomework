import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def k_means(points, n_clusters, max_iter=300, tol=1e-4):
    # 设置若中心点移动距离不大提前停止

    # 将数据转化为浮点数数组
    points = np.array(points, dtype=np.float64)
    # 获取维度以及每个特征的取值范围
    samples, dims = points.shape
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)

    # 随机生成初始中心点
    centers = np.random.uniform(low, high, (n_clusters, dims))

    for i in range(max_iter):
        clusters = np.array([np.argmin(np.linalg.norm(centers - p, axis=1)) for p in points])
        old_centers = np.array(centers)
        for j in range(n_clusters):
            points_in_cluster_j = points[clusters == j]
            centers[j] = np.mean(points_in_cluster_j, axis=0)

        if np.sum(np.linalg.norm(old_centers - centers, axis=1) < tol):
            print(i)
            print(old_centers)
            print(centers)
            break

    return clusters


if __name__ == '__main__':
    data = pd.read_csv(os.path.join('data', 'data.csv'))
    data = data[['difficulty'] + ['Q' + str(i) for i in range(1, 29)]]

    print(k_means(data, n_clusters=3))
