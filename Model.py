from statistics import stdev

from sklearn.feature_selection import RFECV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
import numpy as np
from agregacje import amean  # Arithmetic
from agregacje import qmean  # Quadratic
from scipy.stats import hmean  # Harmonic
from agregacje import gmean  # Geometric


class Model:
    def __init__(self, n_neighbors=3, metric='minkowski', p=None, s=1, t=0.5, aggregation=1, RFECVestimator=None,
                 RFECVstep=1, RFECV_min_n_features=200):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.kNN = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=self.metric, p=self.p)
        self.s = s
        self.t = t
        self.aggregation = aggregation
        self.RFECVestimator = RFECVestimator
        self.RFECVstep = RFECVstep
        self.RFECV_min_n_features = RFECV_min_n_features
        self.rfemodels = None
        self.knnmodels = None
        self.aucs = []
        self.accs = []

    def get_classifier(self):
        return self.kNN

    def get_aggregation(self, number=1):
        aggregations = ['', 'arithmetic', 'quadratic', 'geometric', 'harmonic']
        return aggregations[number]

    def info(self):
        print(f"kNN metric {self.metric} n_neighbors {self.n_neighbors} p {self.p} s {self.s} "
              f"aggregation {self.get_aggregation(number=self.aggregation)} ")

    def get_result(self):
        print(f"Model meanAUC: {np.mean(self.aucs)} stdevAUC: {stdev(self.aucs)} "
              f"meanACC: {np.mean(self.accs)} stdevACC: {stdev(self.accs)}")

    def pred(self, X, y):
        # definiowanie modelu KNN dla k sąsiadów
        rfemodels = []
        knnmodels = []
        for x in range(self.s):  # kłopot z losowanie tym samych tabel z s>1
            selector = RFECV(self.RFECVestimator, min_features_to_select=self.RFECV_min_n_features, step=self.RFECVstep,
                             n_jobs=-1)
            sample = selector.fit_transform(X, y)
            rfemodels.append(selector)
            # print(selector.get_support())
            # tworzenie podtabeli z wybranymi cechami
            # new_X = X[:, sample]
            new_X = sample
            print("tabela:", x + 1, "new_X shape:", new_X.shape, "selected features",
                  [i for i, x in enumerate(selector.support_) if x])
            # budowa klasyfikatora w oparciu o tabelę new_X
            self.kNN.fit(new_X, y)
            knnmodels.append(self.kNN)
        self.knnmodels = knnmodels
        self.rfemodels = rfemodels
        print("Model created")

    def score(self, X, y):
        pred, pimean = self.predict(X)
        self.aucs.append(roc_auc_score(y, pimean))
        self.accs.append(accuracy_score(y, pred))

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

    def result_to_file(self):
        return {'metric': self.metric, 'aggregation': self.get_aggregation(number=self.aggregation),
                'n_neighbors': self.n_neighbors, 's': self.s,
                't': self.t, 'p': self.p, 'RFECV_estimator': self.RFECVestimator,
                'RFECV_min_n_features': self.RFECV_min_n_features,
                'RFECV_step': self.RFECVstep, 'meanAUC': np.mean(self.aucs), 'stdevAUC': stdev(self.aucs),
                'meanACCURACY': np.mean(self.accs), 'stdevACCURACY': stdev(self.accs)}
