import os
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score

warnings.filterwarnings('ignore')


# 获取数据
def get_data(file):
    data = pd.read_csv(file)
    # ID属性不作为输入
    data = np.array(data.drop(['ID'], axis=1), dtype=np.float64)

    np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1]
    y = y.reshape(y.shape[0], 1)
    return X, y


# 进行分类
def xgb_classifier(X, y, pca=False, n_components=0.9, kfolds=5):
    if pca:
        if n_components >= 1:
            n_components = int(n_components)
        X = PCA(n_components=n_components).fit_transform(X)
        print(X.shape)

    clf = XGBClassifier(learning_rate=0.1,
                        n_estimators=154,
                        max_depth=6,
                        min_child_weight=10,
                        gamma=0.,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        objective='binary:logistic',
                        nthread=4,
                        scale_pos_weight=10.0)
    kf = KFold(n_splits=kfolds)
    precision, recall, f2, auc = 0, 0, 0, 0

    # 进行五折交叉验证，将每一次的precision、recall、f2-score、auc取平均作为最终结果
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
        # 输出当前次的结果
        print(output.format(p, r, f, a))

    precision /= kfolds
    recall /= kfolds
    f2 /= kfolds
    auc /= kfolds
    return precision, recall, f2, auc


def main():
    file = os.path.join('data', 'processed_data.csv')
    X, y = get_data(file)
    precision, recall, f2, auc = xgb_classifier(X, y)
    print('Average: \nPrecision: {}\nRecall: {}\nF2-score: {}\nAuc: {}\n'
          .format(precision, recall, f2, auc))


if __name__ == '__main__':
    main()
