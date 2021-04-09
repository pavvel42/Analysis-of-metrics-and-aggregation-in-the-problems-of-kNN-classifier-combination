import numpy as np
import random
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from statistics import stdev
from agregacje import amean  # Arithmetic
from agregacje import qmean  # Quadratic
from scipy.stats import hmean  # Harmonic
from agregacje import gmean  # Geometric


class PredModel:
    def __init__(self, k=3, s=5, t=0.5, aggregation=1, RFECVestimator=None, RFECVstep=1,
                 RFECV_min_n_features=1, metric='minkowski', p=None):
        self.X = None
        self.y = None

        self.k = k
        self.s = s
        self.t = t
        self.aggregation = aggregation
        if RFECVestimator is None:
            RFECVestimator = SVR(kernel="linear")
        self.RFECVestimator = RFECVestimator
        self.RFECVstep = RFECVstep
        self.RFECVcolumnrange = RFECV_min_n_features
        self.metric = metric
        self.p = p

        self.rfemodels = None
        self.knnmodels = None

    def fit(self, X, y):
        self.X = X
        self.y = y

        # pi - tutaj zapisuje wyniki predykcji
        pi = []
        # definiowanie modelu KNN dla k sąsiadów
        rfemodels = []
        knnmodels = []
        for x in range(self.s):
            # losuje liczbę kolumn w zakresie 5 do 100
            # n_cols = random.randint(self.RFEcolumnrange[0], self.RFEcolumnrange[1]) #RFE_n_features=(5, 100)
            # losuje n_cols które kolumny wykorzystam przy budoawniu klasyfikatora
            # sample = random.sample([*range(0, X.shape[1])], k=n_cols)

            # # wybór cech za pomocą RFE - bardzo długo działa ze względu na ilość danych (2000 wierszy 500 atrybutów)
            selector = RFECV(self.RFECVestimator, min_features_to_select=self.RFECVcolumnrange, step=self.RFECVstep, n_jobs=-1)
            sample = selector.fit_transform(X, y)
            rfemodels.append(selector)
            # sample = selector.get_support()
            # print(sample)
            # print(type(sample))
            # tworzenie podtabeli z wybranymi cechami
            # new_X = X[:, sample]
            new_X = sample
            # new_X = selector.transform(X)
            print("tabela:", x, "new_X shape:", new_X.shape, "selected features",
                  [i for i, x in enumerate(selector.support_) if x])

            # budowa klasyfikatora w oparciu o tabelę new_X
            knn = KNeighborsClassifier(n_neighbors=self.k, metric=self.metric, p=self.p)
            knn.fit(new_X, y)
            knnmodels.append(knn)
        self.knnmodels = knnmodels
        self.rfemodels = rfemodels
        print("model created")

    def predict(self, X):
        pi = []
        pimean = None
        for x in range(self.s):
            new_X = self.rfemodels[x].transform(X)
            prob = self.knnmodels[x].predict_proba(new_X)
            pi.append(prob[:, 1])
        pi = np.array(pi)
        if self.aggregation == 1:
            pimean = amean(pi)
        elif self.aggregation == 2:
            pimean = qmean(pi)
        elif self.aggregation == 3:
            pimean = gmean(pi)
        elif self.aggregation == 4:
            pimean = hmean(pi)

        pred = np.where(pimean < self.t, 0, 1)
        return pred, pimean

    def score(self, X, y):
        pred, pimean = self.predict(X)
        auc = roc_auc_score(y, pimean)
        acc = accuracy_score(y, pred)
        print("pimean stdev:", stdev(pimean))
        return auc, acc

    def result(self):
        k = self.k
        s = self.s
        t = self.t
        p = self.p
        aggregation = 'A' + str(self.aggregation)
        metric = self.metric

        if self.aggregation == 1:
            aggregation += ' arithmetic'
        elif self.aggregation == 2:
            aggregation += ' quadratic'
        elif self.aggregation == 3:
            aggregation += ' geometric'
        elif self.aggregation == 4:
            aggregation += ' harmonic'

        RFECVestimator = str(self.RFECVestimator)
        RFECVstep = self.RFECVstep
        RFECVcolumnrange = self.RFECVcolumnrange
        return {'k': k, 's': s, 't': t, 'p': p, 'metric': metric, 'aggregation': aggregation,
                'RFECVestimator': RFECVestimator,
                'RFECVstep': RFECVstep, 'RFECV_min_n_features': RFECVcolumnrange}

    def info(self):
        print(
            f"kNN metric {self.metric} aggregation {self.aggregation} n_neighbors {self.k} k {self.k} s {self.s} p {self.p}")
