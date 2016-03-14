import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold, cross_val_score

obj = load_boston()
X = obj.data
y = obj.target
X = scale(X)
resp, res = -1, -100

for p in np.linspace(1, 10, 200):
    rgs = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p)
    rgs.fit(X, y)
    val = KFold(n=len(y), n_folds=5, shuffle=True, random_state=42)
    now = cross_val_score(X=X, y=y, estimator=rgs, cv=val, scoring='mean_squared_error').mean()

    if now > res:
        res, resp = now, p
print(resp)
