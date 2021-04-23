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
import sys

# sys.stdout = open("Konsola_mniejszy_zestaw.txt", "w")
T = pd.read_csv("D:\GoogleDrive\Dokumenty\Studia\X semestr\Seminarium\Kod\data\colonTumor.csv")
print(T.shape)
scaler = MinMaxScaler()
X = T.iloc[:, 1:T.shape[1]]
X = scaler.fit_transform(X)
y = T.iloc[:, 0]
y = np.where(y == 'negative', 0, 1)
y = LabelEncoder().fit_transform(y)
one_percent = int((T.shape[1] - 1) * 0.01)
# print(one_percent)

metricList = ['euclidean', 'manhattan', 'chebyshev', 'canberra', 'braycurtis']
estimatorSVR = SVR(kernel="linear")
estimatorTree = DecisionTreeClassifier(max_depth=5)
estimators = []
estimators.append(estimatorSVR)
estimators.append(estimatorTree)
kl = [3, 5, 7]
sl = [2, 5, 10, 20]
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
                        Model(n_neighbors=k, metric=metric, s=s, t=0.5, aggregation=i, RFECVestimator=estimator,
                              RFECV_min_n_features=one_percent))
            for i in range(1, 5):
                for p in pl:
                    models.append(
                        Model(n_neighbors=k, metric="minkowski", s=s, t=0.5, aggregation=i, RFECVestimator=estimator,
                              RFECV_min_n_features=one_percent,
                              p=p))
    end = timer()
    print('Time create_models: ', end - start, 'Amount of models: ', len(models), estimator)
    # for model in models:
    #     model.info()
    return models


def pred_models(models_SVR, models_Tree):
    start = timer()
    iter = 0
    samplesSVR = {}
    samplesTree = {}
    for train_index, test_index in skf.split(X, y):
        iter += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # selektor SVR oraz Tree
        selectorSVR = RFECV(estimator=models_SVR[0].get_estimator(), min_features_to_select=one_percent, step=100,
                            n_jobs=-1)
        sample = selectorSVR.fit_transform(X, y)
        selectorTree = RFECV(estimator=models_Tree[0].get_estimator(), min_features_to_select=one_percent, step=100,
                             n_jobs=-1)
        sample = selectorTree.fit_transform(X, y)

        # dict przechowujacy sample dla roznych s
        for s in sl:
            samplesSVR[s] = get_sample(X_train, y_train, n_split=s, selector=selectorSVR, iter=iter,
                                       name_estimator='SVR')
            samplesTree[s] = get_sample(X_train, y_train, n_split=s, selector=selectorTree, iter=iter,
                                        name_estimator='Tree')

        # modele SVR oraz Tree
        for model in models_SVR:
            model.info()
            sampleSVR = get_samples_dict(model=model, samples_dict=samplesSVR)
            for key, val in sampleSVR.items():
                if type(key) is int:
                    model.fit(X_train, y_train, selector=selectorSVR, sample=sampleSVR[key],
                              selected_features=sampleSVR['features_true_' + str(key)])
        for model in models_SVR:
            model.score(X_test, y_test)

        for model in models_Tree:
            model.info()
            sampleTree = get_samples_dict(model=model, samples_dict=samplesTree)
            for key, val in sampleTree.items():
                if type(key) is int:
                    model.fit(X_train, y_train, selector=selectorSVR, sample=sampleTree[key],
                              selected_features=sampleTree['features_true_' + str(key)])
        for model in models_Tree:
            model.score(X_test, y_test)

        print("Finish iteration StratifiedKFold: ", iter)
    end = timer()
    print('Time pred_models: ', end - start)
    for model in models_SVR:
        model.info()
        model.get_result()
        write_to_csv("SVRlinear_Step=100.csv", model.result_to_file())
    for model in models_Tree:
        model.info()
        model.get_result()
        write_to_csv("DecisionTree_Step=100.csv", model.result_to_file())


def get_sample(X, y, n_split, selector, iter, name_estimator):
    samples = {}
    features = dict(enumerate(selector.get_support().flatten(), 0))
    # print(features)
    features_true = {key: val for key, val in features.items() if val == True}
    features_keys = list(features_true.keys())
    random.shuffle(features_keys)
    features_keys = list(split_list(features_keys, n_split))
    # print(features_keys)
    for i in range(len(features_keys)):
        samples[i] = X[:, features_keys[i]]
        samples['features_true_' + str(i)] = features_keys[i]
    print('STATS: StratifiedKFold Iter ', str(iter), 'Estimator ', name_estimator, 'S: ', n_split,
          'count features true ', len(features_true))
    return samples


def get_samples_dict(model, samples_dict):
    if model.get_s() == sl[0]:  # s=2
        samples = samples_dict[2]
    if model.get_s() == sl[1]:  # s=5
        samples = samples_dict[5]
    if model.get_s() == sl[2]:  # s=10
        samples = samples_dict[10]
    if model.get_s() == sl[3]:  # s=20
        samples = samples_dict[20]
    return samples


def split_list(seq, size):
    return (seq[i::size] for i in range(size))


def write_to_csv(filename, result):
    result['Date'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    my_file = Path(filename)
    if not my_file.is_file():
        with open(my_file, 'a') as csvfile:
            fieldnames = ['metric', 'aggregation', 'n_neighbors', 's', 'p', 'RFECV_estimator',
                          'meanAUC', 'stdevAUC', 'meanACCURACY', 'stdevACCURACY', 'stdevACCURACY',
                          'FP_Rate', 'FN_Rate', 'TP_Rate', 'TN_Rate', 'Date']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'metric': 'metric', 'aggregation': 'aggregation', 'n_neighbors': 'n_neighbors',
                             's': 's', 'p': 'p', 'RFECV_estimator': 'RFECV_estimator', 'meanAUC': 'meanAUC',
                             'stdevAUC': 'stdevAUC',
                             'meanACCURACY': 'meanACCURACY', 'stdevACCURACY': 'stdevACCURACY',
                             'FP_Rate': 'FP_Rate', 'FN_Rate': 'FN_Rate', 'TP_Rate': 'TP_Rate', 'TN_Rate': 'TN_Rate',
                             'Date': 'Date'})
    with open(my_file, 'a', newline='') as csvfile:
        fieldnames = ['metric', 'aggregation', 'n_neighbors', 's', 'p', 'RFECV_estimator',
                      'meanAUC', 'stdevAUC', 'meanACCURACY', 'stdevACCURACY', 'stdevACCURACY',
                      'FP_Rate', 'FN_Rate', 'TP_Rate', 'TN_Rate', 'Date']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(result)
    print('Save file: ', filename)


models_SVR = create_models(estimator=estimators[0])
models_Tree = create_models(estimator=estimators[1])
# del models_SVR[1:40]
# del models_Tree[1:40]
pred_models(models_SVR=models_SVR, models_Tree=models_Tree)
# sys.stdout.close()
