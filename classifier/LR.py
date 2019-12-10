import os
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, roc_auc_score

warnings.filterwarnings('ignore')


def get_data(file):
    data = pd.read_csv(file)
    data = np.array(data.drop(['ID'], axis=1), dtype=np.float64)

    np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1]
    y = y.reshape(y.shape[0], 1)
    return X, y


def LR(X, y, pca=False, n_components=0.9, kfolds=5, class_weight=None):
    if pca:
        if n_components >= 1:
            n_components = int(n_components)
        X = PCA(n_components=n_components).fit_transform(X)
        print(X.shape)

    if class_weight is None:
        class_weight = 'balanced'

    clf = LogisticRegression(class_weight=class_weight)
    kf = KFold(n_splits=kfolds)
    precision, recall, f2, auc = 0, 0, 0, 0

    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pro = clf.predict_proba(X_test)[:, -1]
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
    class_weight = {0: 1, 1: 10}
    precision, recall, f2, auc = LR(X, y, class_weight=class_weight)
    print('Average: \nPrecision: {}\nRecall: {}\nF2-score: {}\nAuc: {}\n'
          .format(precision, recall, f2, auc))


if __name__ == '__main__':
    main()

'''
Precision: 0.34109747923606815
Recall: 0.8356327078256125
F2-score: 0.647535641644677
Auc: 0.8883279247219699
'''
