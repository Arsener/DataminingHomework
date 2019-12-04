import os
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def k_means(points, n_clusters, max_iter=300, tol=1e-8):
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
            break


    return clusters


def get_labels(i, data, n_cluster):
    if i == 0:
        return k_means(data, n_clusters=n_cluster)
    else:
        return KMeans(n_clusters=n_cluster, init='random', n_init=10).fit(data).labels_


def calc_sse(data, labels):
    data = np.array(data, dtype=np.float64)
    n_clusters = set(labels)
    centers = np.array([np.mean(data[labels == i], axis=0) for i in n_clusters])
    return sum([np.sum(np.square(np.linalg.norm(data[labels == i] - centers[i], axis=1)))
                for i in n_clusters])


def draw(x, y, y_label):
    colors = ['r', 'b']
    label = ['self implement', 'sklearn']
    for i in range(2):
        plt.plot(x, y[i], marker=".", c=colors[i], label=label[i])

    plt.xlabel("number of clusters")
    plt.ylabel(y_label)

    plt.legend(loc='best')
    plt.savefig(os.path.join('figs', '{}.png'.format(y_label)))
    plt.show()

def main():
    df = pd.read_csv(os.path.join('data', 'data.csv'))
    data = df[['difficulty'] + ['Q' + str(i) for i in range(1, 29)]]

    time_list = [[], []]
    sse_list = [[], []]
    sil_list = [[], []]

    for i in [0, 1]:
        for n_clusters in range(2, 11):
            time_cost = []
            sse = []
            sil = []
            for j in range(10):
                st = time.time()
                labels = get_labels(i, data, n_clusters)
                ed = time.time()
                time_cost.append((ed - st))
                sse.append(calc_sse(data, labels))
                if n_clusters > 1:
                    sil.append(silhouette_score(data, labels, metric='euclidean'))
            time_list[i].append(np.average(time_cost))
            sse_list[i].append(np.average(sse))
            sil_list[i].append(np.average(sil))

    x = range(2, 11)
    draw(x, time_list, 'Time(s)')
    draw(x, sse_list, 'SSE')
    draw(x, sil_list, 'Silhouette Coefficient')

    print(time_list, sse_list, sil_list)


if __name__ == '__main__':
    main()
