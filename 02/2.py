import numpy as np
import pandas
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

data = pandas.read_csv('wine.csv')
X = data[list(range(1, 14))].as_matrix()
y = np.ravel(data[[0]].as_matrix())
ans = [0] * 51

scaler = StandardScaler()
X = scaler.fit_transform(X)

for k in range(1, 51):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)
    val = KFold(n_folds=5, shuffle=True, random_state=42)
    ans[k] = cross_val_score(X=X, y=y, estimator=clf, cv=val).mean()
print(max(ans), ans.index(max(ans)))
