import numpy as np
import pandas
import sklearn.metrics

data = pandas.read_csv('scores.csv')
res, resx = -1, 'false'

for x in ['score_logreg', 'score_svm', 'score_knn', 'score_tree']:
    now = sklearn.metrics.roc_auc_score(data['true'], data[x])

    if now > res:
        res, resx = now, x
print(resx)

res, resx = -1, 'false'

for x in ['score_logreg', 'score_svm', 'score_knn', 'score_tree']:
    prec, rec, thres = sklearn.metrics.precision_recall_curve(data['true'], data[x])
    now = -1

    for i in range(prec.size):
        if rec[i] >= 0.7:
            now = max(now, prec[i])

    if now > res:
        res = now
        resx = x
print(resx)
