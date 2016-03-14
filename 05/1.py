import numpy as np
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold, cross_val_score

data = pandas.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
X = data[list(range(0, len(data.columns) - 1))].as_matrix()
y = np.ravel(data[[-1]].as_matrix())

for k in range(1, 51):
    clf = RandomForestRegressor(n_estimators=k, random_state=1)
    clf.fit(X, y)
    val = KFold(n=len(y), n_folds=5, shuffle=True, random_state=1)
    now = cross_val_score(X=X, y=y, estimator=clf, scoring='r2', cv=val).mean()

    if now >= 0.52:
        print(k)
        exit()
