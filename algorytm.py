from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from Classifier import Classifier
from timeit import default_timer as timer

# pip freeze > requirements.txt

metricList = ['euclidean', 'manhattan', 'chebyshev', 'hamming', 'canberra', 'braycurtis']  # +minkowski wersje
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

T = pd.read_csv("data/colonTumor.csv")
print(T.shape)
scaler = MinMaxScaler()
X = T.iloc[:, 1:T.shape[1]]
X = scaler.fit_transform(X)
y = T.iloc[:, 0]
y = np.where(y == 'negative', 0, 1)
y = LabelEncoder().fit_transform(y)

kNNDict = {}


def algorytm(X, y, k=3, s=5, t=0.5, p=[2]):
    start = timer()
    for metric in metricList:
        kNN = Classifier(n_neighbors=k, metric=metric, p=None)
        kNNDict[metric] = kNN

    for i in p:
        metricList.append('minkowski' + str(i))
        kNN = Classifier(n_neighbors=k, metric='minkowski', p=i)
        kNNDict['minkowski' + str(i)] = kNN
    # print(kNNDict)

    for i in range(s):
        skf = StratifiedKFold(n_splits=10)
        skf.get_n_splits(X, y)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            for metric in metricList:
                kNNDict[metric].fit(X=X_train, y=y_train)
                kNNDict[metric].predict(aggregation='arithmetic', X=X_test, y=y_test, t=t)

    # for metric in metricList:
    # kNNDict[metric].info()
    # kNNDict[metric].aggregation_arithmetic(t=t, y=y)
    # kNNDict[metric].aggregation_quadratic(t=t, y=y)
    # kNNDict[metric].aggregation_harmonic(t=t, y=y)
    # kNNDict[metric].aggregation_geometric(t=t, y=y)
    end = timer()
    print('Czas Wykonania: ', end - start)


algorytm(X=X, y=y, p=[1, 2, 3, 4], s=1)
