import random
from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
import pandas as pd
import numpy as np
from Classifier import Classifier
# pip freeze > requirements.txt

metricList = ['euclidean', 'manhattan', 'chebyshev', 'hamming', 'canberra', 'braycurtis']
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

T = pd.read_csv("data/colonTumor.csv")
print(T.shape)
X = T.iloc[:, 1:T.shape[1]]
y = T.iloc[:, 0]
y = np.where(y == 'negative', 0, 1)
kNNDict = {}


def algorytm(X, y, k=3, s=5, t=0.5, p=[2]):
    for metric in metricList:
        kNN = Classifier(n_neighbors=k, metric=metric, p=None)
        kNNDict[metric] = kNN

    for i in p:
        metricList.append('minkowski' + str(i))
        kNN = Classifier(n_neighbors=k, metric='minkowski', p=i)
        kNNDict['minkowski' + str(i)] = kNN
    # print(kNNDict)

    for i in range(s):
        random_col = random.randint(45, T.shape[0])  # dla testów lokalnych aby przyśpieszyć uzyskanie wyniku na sztywno ustawiam tą wartość min
        print('random_col ', random_col, ' s ', i + 1)
        estimator = SVR(kernel="linear")
        selector = RFECV(estimator, min_features_to_select=random_col, step=1, cv=5, n_jobs=-1).fit_transform(X, y)
        print(selector)
        print(y)
        for metric in metricList:
            kNNDict[metric].fit(selector=selector, y=y)

    for metric in metricList:
        kNNDict[metric].info()
        kNNDict[metric].aggregation_arithmetic(t=t, y=y)
        kNNDict[metric].aggregation_quadratic(t=t, y=y)
        kNNDict[metric].aggregation_harmonic(t=t, y=y)
        kNNDict[metric].aggregation_geometric(t=t, y=y)


algorytm(X=X, y=y, p=[1, 2, 3, 4])
