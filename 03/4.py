import numpy as np
import pandas
import sklearn.metrics

data = pandas.read_csv('classification.csv')
x1 = np.ravel(data['true'].as_matrix())
x2 = np.ravel(data['pred'].as_matrix())
TP, FP, FN, TN = 0, 0, 0, 0

for i in range(x1.size):
    if x1[i] == 1 and x2[i] == 1:
        TP += 1
    elif x1[i] == 1 and x2[i] == 0:
        FN += 1
    elif x1[i] == 0 and x2[i] == 1:
        FP += 1
    else:
        TN += 1

print(TP, FP, FN, TN)
print(sklearn.metrics.accuracy_score(x1, x2), sklearn.metrics.precision_score(x1, x2),
    sklearn.metrics.recall_score(x1, x2), sklearn.metrics.f1_score(x1, x2))
