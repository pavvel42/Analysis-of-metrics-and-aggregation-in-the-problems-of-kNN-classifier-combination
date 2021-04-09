from datetime import datetime
from pathlib import Path
import csv
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from Model import Model
from timeit import default_timer as timer

T = pd.read_csv("D:/GoogleDrive/Dokumenty/Studia/X semestr/Seminarium/Kod/data/colonTumor.csv")
# print(T.shape)
scaler = MinMaxScaler()
X = T.iloc[:, 1:T.shape[1]]
X = scaler.fit_transform(X)
y = T.iloc[:, 0]
y = np.where(y == 'negative', 0, 1)
y = LabelEncoder().fit_transform(y)

# metricList = ['euclidean', 'manhattan', 'chebyshev', 'hamming', 'canberra', 'braycurtis']
metricList = ['euclidean']
models = []
kl = [3]
# sl = [5, 10, 20]
sl = [1]
pl = [1.5, 3, 5, 10, 20]
estimator = SVR(kernel="linear")
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)


def create_models():
    start = timer()
    for k in kl:
        for s in sl:
            for i in range(1, 5):
                for metric in metricList:
                    models.append(
                        Model(n_neighbors=k, metric=metric, s=s, t=0.5, aggregation=i, RFECVestimator=estimator))
            # for i in range(1, 5):
            #     for p in pl:
            #         models.append(Model(n_neighbors=k, metric="minkowski", s=s, t=0.5, aggregation=i, RFECVestimator=estimator, p=p))
    end = timer()
    print('Time create_models: ', end - start, 'Ilosc modeli: ', len(models))
    for model in models:
        model.info()


def pred_models():
    start = timer()
    iter = 1
    for train_index, test_index in skf.split(X, y):
        iter += 1
        count_model = 0
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for model in models:
            count_model += 1
            model.pred(X_train, y_train)
            model.score(X_test, y_test)
            print("Pozostało ", len(models) - count_model, " modeli")
        print("SKF:", iter)
    end = timer()
    print('Time pred_models: ', end - start)
    for model in models:
        model.info()
        model.get_result()
        write_to_csv("SVR_linear.csv", model.result_to_file())


def write_to_csv(filename, result):
    result['Date'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    my_file = Path(filename)
    if not my_file.is_file():
        with open(my_file, 'a') as csvfile:
            fieldnames = ['metric', 'aggregation', 'n_neighbors', 's', 't', 'p', 'RFECV_estimator',
                          'RFECV_min_n_features',
                          'RFECV_step', 'meanAUC', 'stdevAUC', 'meanACCURACY', 'stdevACCURACY', 'stdevACCURACY', 'Date']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'metric': 'metric', 'aggregation': 'aggregation', 'n_neighbors': 'n_neighbors',
                             's': 's', 't': 't', 'p': 'p', 'RFECV_estimator': 'RFECV_estimator',
                             'RFECV_min_n_features': 'RFECV_min_n_features',
                             'RFECV_step': 'RFECV_step', 'meanAUC': 'meanAUC', 'stdevAUC': 'stdevAUC',
                             'meanACCURACY': 'meanACCURACY', 'stdevACCURACY': 'stdevACCURACY', 'Date': 'Date'})
    with open(my_file, 'a', newline='') as csvfile:
        fieldnames = ['metric', 'aggregation', 'n_neighbors', 's', 't', 'p', 'RFECV_estimator', 'RFECV_min_n_features',
                      'RFECV_step', 'meanAUC', 'stdevAUC', 'meanACCURACY', 'stdevACCURACY', 'stdevACCURACY', 'Date']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(result)
    print('Zapisano plik ', filename)


create_models()
pred_models()
