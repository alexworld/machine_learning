import numpy as np
import pandas
from  sklearn.svm import SVC

data = pandas.read_csv('svm-data.csv', header=None)
X = data[[1, 2]].as_matrix()
y = np.ravel(data[[0]].as_matrix())

clf = SVC(kernel='linear', C=100000, random_state=241)
clf.fit(X, y)
print(clf.support_)
