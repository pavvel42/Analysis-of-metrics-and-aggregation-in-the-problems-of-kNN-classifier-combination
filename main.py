from datetime import datetime
import random
from pathlib import Path
import csv
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
from Model import Model
from timeit import default_timer as timer

T = pd.read_csv("D:\GoogleDrive\Dokumenty\Studia\X semestr\Seminarium\Kod\data\colonTumor.csv")
print(T.shape)
scaler = MinMaxScaler()
X = T.iloc[:, 1:T.shape[1]]
X = scaler.fit_transform(X)
y = T.iloc[:, 0]
y = np.where(y == 'negative', 0, 1)
y = LabelEncoder().fit_transform(y)
one_percent = int((T.shape[1]-1)*0.01)
# print(one_percent)

metricList = ['euclidean', 'manhattan', 'chebyshev', 'canberra', 'braycurtis']
estimatorSVR = SVR(kernel="linear")
estimatorTree = DecisionTreeClassifier(max_depth=5)
estimators = []
estimators.append(estimatorSVR)
estimators.append(estimatorTree)
models_all = []
kl = [3, 5, 7]
sl = [20]
pl = [1.5, 3, 5, 10, 20]
# kl = [3]
# sl = [2]
# pl = [1.5]
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(X, y)


def create_models(estimator=None):
    models = []
    start = timer()
    for k in kl:
        for s in sl:
            for i in range(1, 5):
                for metric in metricList:
                    models.append(
                        Model(n_neighbors=k, metric=metric, s=s, t=0.5, aggregation=i, RFECVestimator=estimator, RFECV_min_n_features=one_percent))
            for i in range(1, 5):
                for p in pl:
                    models.append(
                        Model(n_neighbors=k, metric="minkowski", s=s, t=0.5, aggregation=i, RFECVestimator=estimator, RFECV_min_n_features=one_percent,
                              p=p))
    end = timer()
    print('Time create_models: ', end - start, 'Amount of models: ', len(models), estimator)
    # for model in models:
    #     model.info()
    return models


def pred_models(models_SVR, models_Tree):
    start = timer()
    iter = 0
    for train_index, test_index in skf.split(X, y):
        iter += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        selector = RFECV(estimator=models_SVR[0].get_estimator(), min_features_to_select=one_percent, step=100, n_jobs=-1)
        selectorSVR, samplesSVR = get_sample(X_train, y_train, n_split=models_SVR[0].get_s(), selector=selector)
        for model in models_SVR:
            for key, val in samplesSVR.items():
                if type(key) is int:
                    model.fit(X_train, y_train, selector=selectorSVR, sample=samplesSVR[key], selected_features=samplesSVR['features_true_' + str(key)])
        for model in models_SVR:
            model.score(X_test, y_test)

        selector = RFECV(estimator=models_SVR[0].get_estimator(), min_features_to_select=one_percent, step=100, n_jobs=-1)
        selectorTree, samplesTree = get_sample(X_train, y_train, n_split=models_Tree[0].get_s(), selector=selector)
        for model in models_Tree:
            for key, val in samplesTree.items():
                if type(key) is int:
                    model.fit(X_train, y_train, selector=selectorTree, sample=samplesTree[key], selected_features=samplesTree['features_true_' + str(key)])
        for model in models_Tree:
            model.score(X_test, y_test)

        print("Iteration StratifiedKFold: ", iter)
    end = timer()
    print('Time pred_models: ', end - start)
    for model in models_SVR:
        model.info()
        model.get_result()
        write_to_csv("SVR_linear.csv", model.result_to_file())
    for model in models_Tree:
        model.info()
        model.get_result()
        write_to_csv("SVR_linear.csv", model.result_to_file())


def get_sample(X, y, n_split, selector):
    sample = selector.fit_transform(X, y)
    features = dict(enumerate(selector.get_support().flatten(), 0))
    # print(features)
    features_true = {key: val for key, val in features.items() if val == True}
    # print(features_true)
    samples = {}
    for i in range(int(n_split)):
        features_true_copy = features_true.copy()
        for j in range(int(len(list(features_true.keys())) / int(n_split))):
            random_col = random.choice(list(features_true_copy.keys()))
            if i in samples:
                samples[i] = np.append(samples[i], np.array([X[:, random_col]]).T, axis=1)
                samples['features_true_' + str(i)].append(random_col)
            else:
                samples[i] = np.array([X[:, random_col]]).T
                samples['features_true_' + str(i)] = [random_col]
            features_true_copy.pop(random_col)
    return selector, samples,


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
    print('Save file: ', filename)


models_SVR = create_models(estimator=estimators[0])
models_Tree = create_models(estimator=estimators[1])
pred_models(models_SVR=models_SVR, models_Tree=models_Tree)
