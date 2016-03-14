import numpy as np
import pandas
from  sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold, cross_val_score
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )

print('Done')

vect = TfidfVectorizer()
X = vect.fit_transform(raw_documents=newsgroups.data)
y = newsgroups.target

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)

C = gs.best_estimator_.C
print(C)

clf = SVC(kernel='linear', C=C, random_state=241)
clf.fit(X, y)
names = vect.get_feature_names()
resNumber = np.argsort(np.absolute(np.asarray(clf.coef_.todense())).reshape(-1))[-10:]
A = []

for i in resNumber:
    A.append(names[i])

print(' '.join(sorted(A)))
