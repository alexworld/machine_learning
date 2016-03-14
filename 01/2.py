import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas

data = pandas.read_csv('titanic.csv', index_col = 'PassengerId')

X = data.select(lambda col: col in ['Pclass', 'Fare', 'Age', 'Sex'], axis = 1).as_matrix()
y = data.select(lambda col: col == 'Survived', axis = 1).as_matrix()
i = 0

while i < len(X):
    if str(X[i][1]) == 'female':
        X[i][1] = 1
    else:
        X[i][1] = 0
 
    for j in range(4):
       if np.isnan(X[i][j]):
            X = np.delete(X, i, 0)
            y = np.delete(y, i, 0)
            i -= 1
            break
    i += 1
            
clf = DecisionTreeClassifier(random_state = 241)
clf.fit(X, y)

importances = clf.feature_importances_
print(X)
print(importances)
