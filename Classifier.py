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


class Classifier:
    def __init__(self, n_neighbors, metric, p):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.kNN = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=self.metric, p=self.p)
        self.pi = []

    def classifier(self):
        return self.kNN

    def info(self):
        print(f"kNN metric {self.metric} n_neighbors {self.n_neighbors} p {self.p}")

    def fit(self, selector, y):
        self.kNN.fit(selector, y)
        self.pi.append(self.kNN.predict_proba(selector)[:, 0])

    def return_pi(self):
        return self.pi

    def aggregation_arithmetic(self, t, y):
        pi = np.array(self.return_pi())
        agregacja_pi = amean(pi)
        # print("Agregacja arytmetyczna ", agregacja_pi)
        pi = np.where(agregacja_pi > t, 0, 1)
        self.classifier_score(self.aggregation_arithmetic.__name__, pi, y)
        return pi

    def aggregation_quadratic(self, t, y):
        pi = np.array(self.return_pi())
        agregacja_pi = qmean(pi)
        # print("Agregacja kwadratowa ", agregacja_pi)
        pi = np.where(agregacja_pi > t, 0, 1)
        self.classifier_score(self.aggregation_quadratic.__name__, pi, y)
        return pi

    def aggregation_harmonic(self, t, y):
        pi = np.array(self.return_pi())
        agregacja_pi = hmean(pi)
        # print("Agregacja harmoniczna ", agregacja_pi)
        pi = np.where(agregacja_pi > t, 0, 1)
        self.classifier_score(self.aggregation_harmonic.__name__, pi, y)
        return pi

    def aggregation_geometric(self, t, y):
        pi = np.array(self.return_pi())
        agregacja_pi = gmean(pi)
        # print("Agregacja geometryczna ", agregacja_pi)
        pi = np.where(agregacja_pi > t, 0, 1)
        self.classifier_score(self.aggregation_geometric.__name__, pi, y)
        return pi

    def classifier_score(self, method_name, pi, y):
        score = {}
        score[self.metric] = method_name
        score[confusion_matrix.__name__] = self.conf_matrix(pi, y)
        score[self.accuracy.__name__] = self.accuracy(pi, y)
        score[self.precision.__name__] = self.precision(pi, y)
        score[self.recall.__name__] = self.recall(pi, y)
        score[self.f1.__name__] = self.f1(pi, y)
        score[self.auc.__name__] = self.auc(pi, y)
        score[self.roc.__name__] = self.roc(pi, y)
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
