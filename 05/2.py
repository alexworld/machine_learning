import numpy as np
import pandas
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

data = pandas.read_csv('gbm-data.csv')
X = data[list(range(1, len(data.columns)))]
y = np.ravel(data[[0]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

for learning_rate in [1, 0.5, 0.3, 0.2, 0.1]:
    clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=learning_rate)
    clf.fit(X_train, y_train)

    log_train = []
    log_test = []
    
    for y_pred in clf.staged_decision_function(X_train):
        log_train.append(log_loss(y_train, 1 / (1 + np.exp(-y_pred))))

    for y_pred in clf.staged_decision_function(X_test):
        log_test.append(log_loss(y_test, 1 / (1 + np.exp(-y_pred))))
    
    if learning_rate == 0.2:
        mini = min(log_test)
        ind = (log_test).index(mini)
    
    plt.figure()
    plt.plot(log_test, 'r', linewidth=2)
    plt.plot(log_train, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()

print()
print(mini, ind)

clf = RandomForestClassifier(n_estimators=ind, random_state=241)
clf.fit(X_train, y_train)
print(log_loss(y_test, clf.predict_proba(X_test)[:, 1]))
