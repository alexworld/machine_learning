import numpy as np
import pandas
import sklearn.metrics
import math

def grad(x1, x2, y, C):
    k = 0.1
    w1p, w2p = -1, -1
    w1, w2 = 0, 0

    while math.sqrt((w1 - w1p) ** 2 + (w2 - w2p) ** 2) > 1e-5:
        w1p, w2p = w1, w2

        now = 0

        for i in range(x1.size):
            now += y[i] * x1[i] * (1 - 1 / (1 + pow(math.e, -y[i] * (w1 * x1[i] + w2 * x2[i]))))
        w1 = w1 + k / x1.size * now - k * C * w1

        now = 0

        for i in range(x1.size):
            now += y[i] * x2[i] * (1 - 1 / (1 + pow(math.e, -y[i] * (w1 * x1[i] + w2 * x2[i]))))
        w2 = w2 + k / x1.size * now - k * C * w2

    score = [0] * x1.size

    for i in range(x1.size):
        score[i] = 1 / (1 + pow(math.e, -w1 * x1[i] - w2 * x2[i]))
    return sklearn.metrics.roc_auc_score(y, score)

data = pandas.read_csv('data-logistic.csv')
y = np.ravel(data[[0]].as_matrix())
x1 = np.ravel(data[[1]].as_matrix())
x2 = np.ravel(data[[2]].as_matrix())
print(grad(x1, x2, y, 0), grad(x1, x2, y, 10))
