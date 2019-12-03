import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def compare_cluster_results(data, max_clusters=10, pca=False, n_components=0.9):
    if pca:
        data = PCA(n_components=n_components).fit_transform(data)

    # 记录不同类数的轮廓系数结果
    x_label_silhouette_score = []
    y_label_silhouette_score = []
    for n_clusters in range(2, max_clusters):
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit(data)
        # 计算轮廓系数
        silhouette_avg = silhouette_score(data, model.labels_, metric='euclidean')
        x_label_silhouette_score.append(n_clusters)
        y_label_silhouette_score.append(silhouette_avg)

    plt.plot(x_label_silhouette_score, y_label_silhouette_score, marker="o")
    plt.xlabel("The number of clusters")
    plt.ylabel("Silhouette coefficient")
    # plt.savefig(os.path.join('figs', 'hierarchy_sil_pca_{}.png'.format(str(pca))))
    plt.show()


def get_cluster_result(data, n_clusters=2, pca=False, n_components=0.9):
    if pca:
        data = PCA(n_components=n_components).fit_transform(data)

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit(data)
    return model


def visualized(data, labels, c):
    tsne = TSNE(n_components=2, metric='euclidean', init='pca')
    tsne_2d = tsne.fit_transform(data)

    for i in range(len(c)):
        cluster_i = tsne_2d[[l[0] for l in np.argwhere(labels == i)]]
        plt.scatter(cluster_i[:, 0], cluster_i[:, 1], c=c[i], marker='.')
    plt.savefig(os.path.join('figs', 'hierarchy_visualization.png'))
    plt.show()


def main():
    # 获取数据
    df = pd.read_csv(os.path.join('data', 'data.csv'))
    # 选择输入的字段：课程的难度difficulty以及28个问题Q1-Q28
    data = df[['difficulty'] + ['Q' + str(i) for i in range(1, 29)]]
    compare_cluster_results(data, pca=True)

    model = get_cluster_result(data, n_clusters=2, pca=True)
    # visualized(data, model.labels_, c=['r', 'b'])


if __name__ == '__main__':
    main()
