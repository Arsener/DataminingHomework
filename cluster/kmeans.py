import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# 比较簇数量不同时以及是否降维时的SSE和轮廓系数。默认设置PCA的参数n_components=0.9，即保留90%的信息
def compare_cluster_results(data, max_clusters=10, n_components=0.9):
    # 记录不同类数的SSE和轮廓系数结果
    x_label_silhouette_score = [[], []]
    y_label_silhouette_score = [[], []]
    x_label_SSE = [[], []]
    y_label_SSE = [[], []]

    for i in range(2):
        if i == 1:
            if n_components >= 1:
                n_components = int(n_components)
            data = PCA(n_components=n_components).fit_transform(data)

        model = KMeans(n_clusters=1, init='random', n_init=10).fit(data)
        x_label_SSE[i].append(1)
        y_label_SSE[i].append(model.inertia_)
        for n_clusters in range(2, max_clusters + 1):
            model = KMeans(n_clusters=n_clusters, init='random', n_init=10).fit(data)
            # 计算轮廓系数
            silhouette_avg = silhouette_score(data, model.labels_, metric='euclidean')
            x_label_silhouette_score[i].append(n_clusters)
            y_label_silhouette_score[i].append(silhouette_avg)
            x_label_SSE[i].append(n_clusters)
            y_label_SSE[i].append(model.inertia_)

    plt.plot(x_label_SSE[0], y_label_SSE[0], marker="o", c='r', label='without PCA')
    plt.plot(x_label_SSE[1], y_label_SSE[1], marker="o", c='b', label='with PCA')
    plt.xlabel("The number of clusters")
    plt.ylabel("SSE")
    plt.legend(loc='best')
    plt.savefig(os.path.join('figs', 'kmeans_sse.png'))
    plt.show()
    plt.plot(x_label_silhouette_score[0], y_label_silhouette_score[0], marker="o", c='r', label='without PCA')
    plt.plot(x_label_silhouette_score[1], y_label_silhouette_score[1], marker="o", c='b', label='with PCA')
    plt.xlabel("The number of clusters")
    plt.ylabel("Silhouette coefficient")
    plt.legend(loc='best')
    plt.savefig(os.path.join('figs', 'kmeans_sil.png'))
    plt.show()


# 得到聚类结果以及用于聚类的数据
def get_cluster_result(data, n_clusters=2, pca=False, n_components=0.9):
    if pca:
        if n_components >= 1:
            n_components = int(n_components)
        data = PCA(n_components=n_components).fit_transform(data)

    model = KMeans(n_clusters=n_clusters, init='random', n_init=10).fit(data)
    return model, data


# 可视化
def visualized(data, labels, c):
    tsne = TSNE(n_components=2, metric='euclidean', init='pca')
    tsne_2d = tsne.fit_transform(data)

    for i in range(len(c)):
        cluster_i = tsne_2d[[l[0] for l in np.argwhere(labels == i)]]
        plt.scatter(cluster_i[:, 0], cluster_i[:, 1], c=c[i], marker='.')
    plt.savefig(os.path.join('figs', 'kmeans_visualization.png'))
    plt.show()


# 进行数据分析
def data_analysis(df, labels, data, x, colors):
    # 添加一列，内容为：教师编号-课程编号
    def combine(a, b):
        return str(a) + '-' + str(b)

    df['instr-class'] = list(map(lambda a, b: combine(a, b), df['instr'], df['class']))
    # 统计每一簇中每个问题的平均分数，并绘出折线图
    plt.figure(figsize=(13, 4))
    for i in range(3):
        plt.plot(x, np.mean(np.array(data[labels == i]), axis=0),
                 marker=".", c=colors[i], label='cluster{}'.format(i))

    plt.xlabel("Question")
    plt.ylabel("Average score")
    plt.legend(loc='best')
    plt.savefig(os.path.join('figs', 'average_score_in_different_clusters.png'))
    plt.show()

    # 统计每个老师（课程）的学生在每个簇中占的比例
    for name in ['instr', 'instr-class']:
        pd_instr = pd.DataFrame(
            [[key] + [dict(df[labels == i][name].value_counts())[key] / value for i in range(3)] for key, value
             in
             dict(df[name].value_counts()).items()])
        pd_instr.columns = [name, 'clsuter0', 'cluster1', 'cluster2']
        pd_instr.to_csv(os.path.join('data', '{}_result.csv'.format(name)), index=False)


# 计算SSE
def calc_sse(data, labels):
    data = np.array(data, dtype=np.float64)
    n_clusters = set(labels)
    centers = np.array([np.mean(data[labels == i], axis=0) for i in n_clusters])
    return sum([np.sum(np.square(np.linalg.norm(data[labels == i] - centers[i], axis=1)))
                for i in n_clusters])


def main():
    # 获取数据
    df = pd.read_csv(os.path.join('data', 'data.csv'))
    # 选择输入的字段：课程的难度difficulty以及28个问题Q1-Q28
    x = ['difficulty'] + ['Q' + str(i) for i in range(1, 29)]
    data = df[x]
    # compare_cluster_results(data)

    model, data_for_cluster = get_cluster_result(data, n_clusters=3, pca=True)

    colors = ['r', 'g', 'b']
    silhouette_avg = silhouette_score(data_for_cluster, model.labels_, metric='euclidean')
    sse = calc_sse(data_for_cluster, model.labels_)
    print('Silhouette Coefficient: {}\nSSE: {}'.format(silhouette_avg, sse))
    # visualized(data_for_cluster, model.labels_, c=colors)
    # data_analysis(df, model.labels_, data, x, colors)


if __name__ == '__main__':
    main()
