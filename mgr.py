from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
import pandas as pd
import numpy as np
import random
from agregacje import amean  # Arithmetic
from agregacje import qmean  # Quadratic
from scipy.stats import hmean  # Harmonic
from scipy.stats import gmean  # Geometric

# tablica_decyzyjna [T] = (U,A,d)
# decyzja [A]
# obiekt_testowy [u]?
metricList = (
    'euclidean', 'manhattan', 'chebyshev', 'wminkowski', 'seuclidean', 'mahalanobis', 'hamming', 'canberra',
    'braycurtis')

T = pd.read_csv("data/colonTumor.csv")
print(T.shape)
X = T.iloc[:, 1:T.shape[1]]
y = T.iloc[:, 0]
y = np.where(y == 'negative', 0, 1)

def algorytm(X, y, k=3, s=5, t=0.5, metric='euclidean', p=2):
    pi = []
    kNN = KNeighborsClassifier(n_neighbors=k, metric=metric, p=p)
    kNN.fit(X, y)  # X = selector

    for i in range(s):
        # random_col = random.randint(1, T.shape[0])
        random_col = 61
        print('random_col ', random_col, ' s ', i, ' metric ', metric, ' p ', p)

        estimator = SVR(kernel="linear")
        selector = RFECV(estimator, min_features_to_select=random_col, step=1, cv=3).fit_transform(X, y)

        kNN.fit(selector, y)  # X = selector

        proba = kNN.predict_proba(selector)
        pi.append(proba[:, 0])
        # print(proba)
    pi = np.array(pi)
    # print(pi)
    agregacja_pi = qmean(pi)
    print("Agregacja pi ", agregacja_pi)
    pi = np.where(agregacja_pi > t, 0, 1)
    print("if p > t then ", pi)


algorytm(X=X, y=y, metric='euclidean', p=3)


