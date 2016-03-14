import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.preprocessing
import pandas

def learn_and_check(scaled):
    data = pandas.read_csv('perceptron-train.csv')
    X = data[[1, 2]].as_matrix()
    y = np.ravel(data[[0]].as_matrix())

    if scaled:
        scaler = sklearn.preprocessing.StandardScaler()
        X = scaler.fit_transform(X)

    clf = sklearn.linear_model.Perceptron(random_state = 241)
    clf.fit(X, y)

    data = pandas.read_csv('perceptron-test.csv')
    X = data[[1, 2]].as_matrix()
    y = np.ravel(data[[0]].as_matrix())

    if scaled:
        X = scaler.transform(X)

    pred = clf.predict(X)
    return sklearn.metrics.accuracy_score(y, pred)

print(learn_and_check(True) - learn_and_check(False))
