from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
import numpy as np
import random
import pandas as pd
from statistics import stdev
from agregacje import amean  # Arithmetic
from agregacje import qmean  # Quadratic
from scipy.stats import hmean  # Harmonic
from agregacje import gmean  # Geometric


class Classifier:
    def __init__(self, n_neighbors, metric, p):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.kNN = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=self.metric, p=self.p)
        self.pi = []
        self.aucs = []
        self.accs = []

    def classifier(self):
        return self.kNN

    def info(self):
        print(f"kNN metric {self.metric} n_neighbors {self.n_neighbors} p {self.p}")

    def fit(self, X, y):
        random_col = random.randint(1,
                                    y.size)  # dla testów lokalnych aby przyśpieszyć uzyskanie wyniku na sztywno ustawiam tą wartość min
        # print('random_col ', random_col, ' s ', i + 1)
        estimator = SVR(kernel="linear")
        selector = RFECV(estimator, min_features_to_select=random_col, step=1, cv=5, n_jobs=-1).fit_transform(X, y)
        # print(selector)
        # print(y)
        self.kNN.fit(selector, y)
        #self.pi.append(self.kNN.predict_proba(selector)[:, 0])
        # self.aucs.append(self.auc(self.pi, y))
        # self.accs.append(self.accuracy(self.pi, y))

    def return_pi(self):
        return self.pi

    def predict(self, X, y, t, aggregation='arithmetic'):
        prob = self.kNN.predict_proba(X)
        self.pi.append(prob[:, 1])
        pi = np.array(self.return_pi())
        if aggregation == 'arithmetic':
            pimean = amean(pi)
        elif aggregation == 'quadratic':
            pimean = qmean(pi)
        elif aggregation == 'harmonic':
            pimean = hmean(pi)
        elif aggregation == 'geometric':
            pimean = gmean(pi)
        pred = np.where(pimean > t, 0, 1)
        self.aucs.append(self.auc(pi=pimean, y=y))
        self.accs.append(self.accuracy(pi=pred, y=y))
        self.classifier_score(aggregation)

    def classifier_score(self, method_name, pi, y):
        score = {}
        score[self.metric] = method_name
        # score[confusion_matrix.__name__] = self.conf_matrix(pi, y)
        # score[self.accuracy.__name__] = self.accuracy(pi, y)
        # score[self.precision.__name__] = self.precision(pi, y)
        # score[self.recall.__name__] = self.recall(pi, y)
        # score[self.f1.__name__] = self.f1(pi, y)
        # score[self.auc.__name__] = self.auc(pi, y)
        # score[self.roc.__name__] = self.roc(pi, y)
        score[self.aucs.__name__] = np.mean(self.aucs)
        score[self.accs.__name__] = stdev(self.aucs)
        print(score)

    def conf_matrix(self, pi, y):
        tn, fp, fn, tp = confusion_matrix(y, pi).ravel()
        return tn, fp, fn, tp

    def accuracy(self, pi, y):
        return accuracy_score(y, pi)

    def precision(self, pi, y):
        return precision_score(y, pi)

    def recall(self, pi, y):
        return recall_score(y, pi)

    def f1(self, pi, y):
        return f1_score(y, pi)

    def auc(self, pi, y):
        return roc_auc_score(y, pi)

    def roc(self, pi, y):
        return roc_curve(y, pi, pos_label=1)
