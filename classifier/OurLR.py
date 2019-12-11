import os
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import time

warnings.filterwarnings('ignore')


def get_data(file):
    data = pd.read_csv(file)
    data = np.array(data.drop(['ID'], axis=1), dtype=np.float64)

    np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1]
    y = y.reshape(y.shape[0], 1)
    return X, y


class LogisticRegression():
    """
        Parameters:
        -----------
        n_iterations: int
            梯度下降的轮数
        learning_rate: float
            梯度下降学习率
    """

    def __init__(self, learning_rate=.1, n_iterations=4000, class_weight=1.):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.class_weight = class_weight

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def initialize_weights(self, n_features):
        # 初始化参数
        # 参数范围[-1/sqrt(N), 1/sqrt(N)]
        limit = np.sqrt(1 / n_features)
        w = np.random.uniform(-limit, limit, (n_features, 1))
        b = 0
        self.w = np.insert(w, 0, b, axis=0)

    def fit(self, X, y):
        m_samples, n_features = X.shape
        self.initialize_weights(n_features)
        # 为X增加一列特征x1，x1 = 0
        X = np.insert(X, 0, 1, axis=1)
        y = np.reshape(y, (m_samples, 1))

        # 梯度训练n_iterations轮
        for i in range(self.n_iterations):
            h_x = X.dot(self.w)
            y_pred = self.sigmoid(h_x)
            y = y.reshape((y.shape[0],))
            y_pred = y_pred.reshape((y.shape[0],))
            error = y_pred - y

            # 找到预测为负例的正例，并将损失乘上权重
            mask_a = np.ones((y.shape[0],), dtype=np.bool)
            mask_b = np.ones((y.shape[0],), dtype=np.bool)
            mask_a[np.argwhere(y == 0)] = False
            mask_b[np.argwhere(y_pred > 0.5)] = False
            error[mask_a & mask_b] *= self.class_weight

            error = error.reshape((y.shape[0], 1))
            y = y.reshape((y.shape[0], 1))

            w_grad = X.T.dot(error)
            self.w = self.w - self.learning_rate * w_grad

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        h_x = X.dot(self.w)
        y_prob = self.sigmoid(h_x)
        y_pred = np.round(y_prob).astype(int)
        return y_pred, y_prob


def LR(X, y, pca=False, n_components=0.9, kfolds=5, class_weight=1.):
    if pca:
        if n_components >= 1:
            n_components = int(n_components)
        X = PCA(n_components=n_components).fit_transform(X)
        print(X.shape)

    clf = LogisticRegression(class_weight=class_weight)
    kf = KFold(n_splits=kfolds)
    precision, recall, f2, auc = 0, 0, 0, 0

    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        startTime = time.time()
        clf.fit(X_train, y_train)
        print('Congratulations, training complete! Took %fs!' % (time.time() - startTime))

        y_pred, y_pro = clf.predict(X_test)
        p = precision_score(y_test, y_pred)
        r = recall_score(y_test, y_pred)
        f = 5 * p * r / (4 * p + r)
        a = roc_auc_score(y_test, y_pro)
        output = 'Precision: {}\nRecall: {}\nF2-score: {}\nAuc: {}\n'
        precision += p
        recall += r
        f2 += f
        auc += a

        print(output.format(p, r, f, a))

    precision /= kfolds
    recall /= kfolds
    f2 /= kfolds
    auc /= kfolds
    return precision, recall, f2, auc


def main():
    file = os.path.join('data', 'processed_data.csv')
    X, y = get_data(file)
    precision, recall, f2, auc = LR(X, y, class_weight=9.)
    print('Average: \nPrecision: {}\nRecall: {}\nF2-score: {}\nAuc: {}\n'
          .format(precision, recall, f2, auc))


if __name__ == '__main__':
    main()

''' 
Precision: 0.3352612208229166
Recall: 0.7884159517341666
F2-score: 0.5218788005506723
Auc: 0.7252021099809538
'''
