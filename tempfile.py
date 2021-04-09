from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from testclass import TestClass
from predmodel import PredModel
from timeit import default_timer as timer

T = pd.read_csv("D:/GoogleDrive/Dokumenty/Studia/X semestr/Seminarium/Kod/data/colonTumor.csv")
# print(T.shape)
scaler = MinMaxScaler()
X = T.iloc[:, 1:T.shape[1]]
X = scaler.fit_transform(X)
y = T.iloc[:, 0]
y = np.where(y == 'negative', 0, 1)
y = LabelEncoder().fit_transform(y)

filename = "SVRlinear"
# savepath = "../xls/" + filename + ".csv"
savepath = filename + ".csv"

# metricList = ['euclidean', 'manhattan', 'chebyshev', 'hamming', 'canberra', 'braycurtis']  # + 'minkowski' z param p
metricList = ['euclidean']
models = []

# kl = [3, 5, 7]
# sl = [5, 10, 20]
# pl = [1.5, 3, 5, 10, 20]
kl = [3]
sl = [1]
pl = [1.5]
start = timer()
for k in kl:
    for s in sl:
        for i in range(1, 2):
            for metric in metricList:
                models.append(
                    PredModel(k=k, s=s, t=0.5, aggregation=i, RFECVestimator=None, RFECV_min_n_features=1,
                              metric=metric))
        # for i in range(1, 4):
        #     for p in pl:
        #         models.append(
        #             PredModel(k=k, s=s, t=0.5, aggregation=i, RFECVestimator=None, RFECV_min_n_features=1,
        #                       metric='minkowski', p=p))
end = timer()
print('Czas Wykonania: ', end - start)

start = timer()
for model in models:
    model.info()
    test = TestClass(X, y, model, savepath)
end = timer()
print('Czas Wykonania: ', end - start)

# estimator = DecisionTreeClassifier(max_depth=5)
