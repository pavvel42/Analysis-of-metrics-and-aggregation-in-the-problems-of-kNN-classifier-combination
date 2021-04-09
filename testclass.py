import csv
from pathlib import Path
from statistics import stdev

from IPython.display import clear_output
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
import numpy as np


class TestClass:
    def __init__(self, X, y, PredModel, filename="document.csv"):
        skf = StratifiedKFold(n_splits=10)
        skf.get_n_splits(X, y)

        test = PredModel
        aucs = []
        accs = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            test.fit(X_train, y_train)
            auc, acc = test.score(X_test, y_test)
            aucs.append(auc)
            accs.append(acc)
            print("FOLD acc:", acc, "auc", auc)

        result = test.result()
        result['meanAUC'] = np.mean(aucs)
        result['stdevAUC'] = stdev(aucs)

        result['meanACCURACY'] = np.mean(accs)
        result['stdevACCURACY'] = stdev(accs)
        result['Date'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        # ['k','s','t','aggregation','aparam','RFEestimator','RFEstep',
        # 'RFE_n_features','meanAUC','stdevAUC','meanACCURACY','stdevACCURACY']

        my_file = Path(filename)
        if not my_file.is_file():
            with open(my_file, 'a') as csvfile:
                fieldnames = ['k', 's', 't', 'p', 'metric', 'aggregation', 'RFECVestimator', 'RFECVstep', 'RFECV_min_n_features',
                              'meanAUC', 'stdevAUC', 'meanACCURACY', 'stdevACCURACY', 'Date']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'k': 'k', 's': 's', 't': 't', 'p': 'p', 'metric': 'metric', 'aggregation': 'aggregation',
                                 'RFECVestimator': 'RFECVestimator', 'RFECVstep': 'RFECVstep',
                                 'RFECV_min_n_features': 'RFECV_min_n_features', 'meanAUC': 'meanAUC', 'stdevAUC': 'stdevAUC',
                                 'meanACCURACY': 'meanACCURACY', 'stdevACCURACY': 'stdevACCURACY', 'Date': 'Date'})

        with open(my_file, 'a', newline='') as csvfile:
            fieldnames = ['k', 's', 't', 'p', 'metric', 'aggregation', 'RFECVestimator', 'RFECVstep', 'RFECV_min_n_features', 'meanAUC',
                          'stdevAUC', 'meanACCURACY', 'stdevACCURACY', 'Date']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(result)
        clear_output()
        print('zapisano plik')
