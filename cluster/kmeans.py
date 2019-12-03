import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compare_cluster_results(data, max_clusters=10, pca=False, n_components=0.9):
    if pca:
        data = PCA(n_components=n_components).fit_transform(data)

    # 记录不同类数的轮廓系数结果
    x_label_silhouette_score = []
    y_label_silhouette_score = []

    model = KMeans(n_clusters=1, init='random', n_init=10).fit(data)
    x_label_SSE = [1]
    y_label_SSE = [model.inertia_]
    for n_clusters in range(2, max_clusters):
        model = KMeans(n_clusters=n_clusters, init='random', n_init=10).fit(data)
        # 计算轮廓系数
        silhouette_avg = silhouette_score(data, model.labels_, metric='euclidean')
        x_label_silhouette_score.append(n_clusters)
        y_label_silhouette_score.append(silhouette_avg)
        x_label_SSE.append(n_clusters)
        y_label_SSE.append(model.inertia_)

    plt.plot(x_label_SSE, y_label_SSE, marker="o")
    plt.xlabel("The number of clusters")
    plt.ylabel("SSE")
    # plt.savefig(os.path.join('figs', 'kmeans_sse_pca_{}.png'.format(str(pca))))
    plt.show()
    plt.plot(x_label_silhouette_score, y_label_silhouette_score, marker="o")
    plt.xlabel("The number of clusters")
    plt.ylabel("Silhouette coefficient")
    # plt.savefig(os.path.join('figs', 'kmeans_sil_pca_{}.png'.format(str(pca))))
    plt.show()


def get_cluster_result(data, n_clusters=2, pca=False, n_components=0.9):
    if pca:
        data = PCA(n_components=n_components).fit_transform(data)

    model = KMeans(n_clusters=n_clusters, init='random', n_init=10).fit(data)
    return model


def visualized(data, labels, c):
    tsne = TSNE(n_components=2, metric='euclidean', init='pca')
    tsne_2d = tsne.fit_transform(data)

    for i in range(len(c)):
        cluster_i = tsne_2d[[l[0] for l in np.argwhere(labels == i)]]
        plt.scatter(cluster_i[:, 0], cluster_i[:, 1], c=c[i], marker='.')
    plt.savefig(os.path.join('figs', 'kmeans_visualization.png'))
    plt.show()


def combine(a, b):
    return str(a) + '-' + str(b)


def main():
    colors = ['r', 'g', 'b']
    # 获取数据
    df = pd.read_csv(os.path.join('data', 'data.csv'))
    # 选择输入的字段：课程的难度difficulty以及28个问题Q1-Q28
    x = ['difficulty'] + ['Q' + str(i) for i in range(1, 29)]
    data = df[x]
    # compare_cluster_results(data, pca=True)

    model = get_cluster_result(data, n_clusters=3, pca=True)
    # visualized(data, model.labels_, c=colors)

    df['instr-class'] = list(map(lambda a, b: combine(a, b), df['instr'], df['class']))
    plt.figure(figsize=(13, 4))
    for i in range(3):
        plt.plot(x, np.mean(np.array(data[model.labels_ == i]), axis=0),
                 marker=".", c=colors[i], label='cluster{}'.format(i))

    plt.xlabel("Question")
    plt.ylabel("Average score")

    plt.legend(loc='best')
    # plt.savefig(os.path.join('figs', 'average_score_in_different_clusters.png'))
    plt.show()

    for name in ['instr', 'instr-class']:
        pd_instr = pd.DataFrame(
            [[key] + [dict(df[model.labels_ == i][name].value_counts())[key] / value for i in range(3)] for key, value in
             dict(df[name].value_counts()).items()])
        pd_instr.columns = [name, 'clsuter0', 'cluster1', 'cluster2']
        # pd_instr.to_csv(os.path.join('data', '{}_result.csv'.format(name)), index=False)



if __name__ == '__main__':
    main()

