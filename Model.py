import copy
from statistics import stdev
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import numpy as np
from agregacje import amean  # Arithmetic
from agregacje import qmean  # Quadratic
from scipy.stats import hmean  # Harmonic
from agregacje import gmean  # Geometric


class Model:
    def __init__(self, n_neighbors=3, metric='minkowski', p=None, s=2, t=0.5, aggregation=1, RFECVestimator=None,
                 RFECVstep=100, RFECV_min_n_features=200):
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
        self.rfemodels = []
        self.knnmodels = []
        self.aucs = []
        self.accs = []
        self.conf_matrix = {}

    def get_classifier(self):
        return self.kNN

    def get_estimator(self):
        return self.RFECVestimator

    def get_s(self):
        return self.s

    def get_aggregation(self, number=1):
        aggregations = ['', 'arithmetic', 'quadratic', 'geometric', 'harmonic']
        return aggregations[number]

    def info(self):
        print(f"kNN metric {self.metric} n_neighbors {self.n_neighbors} p {self.p} s {self.s} "
              f"aggregation {self.get_aggregation(number=self.aggregation)} ")

    def get_result(self):
        print(f"Model meanAUC: {np.mean(self.aucs)} stdevAUC: {stdev(self.aucs)} "
              f"meanACC: {np.mean(self.accs)} stdevACC: {stdev(self.accs)}")

    def fit(self, X, y, selector, sample, selected_features):
        # print(sample)
        new_X = sample
        # rfemodels.append(selector)
        print('count features ', len(selected_features), 'selected_features ', selected_features)

        copy_knn = copy.deepcopy(self.kNN)
        copy_knn.fit(new_X, y)
        self.knnmodels.append(copy_knn)
        self.rfemodels.append(selected_features)

    def score(self, X, y):
        pred, pimean = self.predict(X)
        self.aucs.append(roc_auc_score(y, pimean))
        self.accs.append(accuracy_score(y, pred))
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        # print(tn, fp, fn, tp)
        self.conf_matrix['tn'] = self.conf_matrix.get('tn', 0) + tn
        self.conf_matrix['fp'] = self.conf_matrix.get('fp', 0) + fp
        self.conf_matrix['fn'] = self.conf_matrix.get('fn', 0) + fn
        self.conf_matrix['tp'] = self.conf_matrix.get('tp', 0) + tp

    def predict(self, X):
        pi = []
        pimean = None

        for x in range(len(self.knnmodels)):
            new_X = X[:, self.rfemodels[x]]
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
        FP_Rate = self.conf_matrix['fp'] / (self.conf_matrix['fp'] + self.conf_matrix['tn'])
        FN_Rate = self.conf_matrix['fn'] / (self.conf_matrix['tp'] + self.conf_matrix['fn'])
        TP_Rate = self.conf_matrix['tp'] / (self.conf_matrix['tp'] + self.conf_matrix['fn'])
        TN_Rate = self.conf_matrix['tn'] / (self.conf_matrix['fp'] + self.conf_matrix['tn'])
        # print(FP_Rate, FN_Rate, TP_Rate, TN_Rate)
        return {'metric': self.metric, 'aggregation': self.get_aggregation(number=self.aggregation),
                'n_neighbors': self.n_neighbors, 's': self.s,
                'p': self.p, 'RFECV_estimator': self.RFECVestimator,
                'meanAUC': np.mean(self.aucs), 'stdevAUC': stdev(self.aucs),
                'meanACCURACY': np.mean(self.accs), 'stdevACCURACY': stdev(self.accs),
                'FP_Rate': FP_Rate, 'FN_Rate': FN_Rate, 'TP_Rate': TP_Rate, 'TN_Rate': TN_Rate}
