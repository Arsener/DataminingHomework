import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def k_means(points, n_clusters, max_iter=300, tol=1e-8):
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
        # 计算新的中心点
        for j in range(n_clusters):
            points_in_cluster_j = points[clusters == j]
            centers[j] = np.mean(points_in_cluster_j, axis=0)

        # 若中心点位置变化不大则提前停止
        if np.sum(np.linalg.norm(old_centers - centers, axis=1) < tol):
            break

    return clusters


# 使用两种不同的kmeans算法得到聚类结果
def get_labels(i, data, n_cluster):
    if i == 0:
        return k_means(data, n_clusters=n_cluster)
    else:
        return KMeans(n_clusters=n_cluster, init='random', n_init=10).fit(data).labels_


# 计算SSE
def calc_sse(data, labels):
    data = np.array(data, dtype=np.float64)
    n_clusters = set(labels)
    centers = np.array([np.mean(data[labels == i], axis=0) for i in n_clusters])
    return sum([np.sum(np.square(np.linalg.norm(data[labels == i] - centers[i], axis=1)))
                for i in n_clusters])


# 画出折线图
def draw(x, y, y_label):
    colors = ['r', 'b']
    label = ['self implement', 'sklearn']
    plt.figure(figsize=(3.5, 2.5))
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

    # 分别记录两种算法的聚类结果相关信息
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
                # 计算耗时
                time_cost.append((ed - st))
                sse.append(calc_sse(data, labels))
                sil.append(silhouette_score(data, labels, metric='euclidean'))
            # 计算十次的平均值
            time_list[i].append(np.average(time_cost))
            sse_list[i].append(np.average(sse))
            sil_list[i].append(np.average(sil))

    x = range(2, 11)
    draw(x, time_list, 'Time(s)')
    draw(x, sse_list, 'SSE')
    draw(x, sil_list, 'Silhouette Coefficient')


if __name__ == '__main__':
    main()
