import numpy as np
import pandas
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack
from sklearn.decomposition import PCA

data = pandas.read_csv('close_prices.csv', header=None)
X = data.drop([0], axis=1)
X = X.drop([0], axis=0)

res = PCA(n_components=0.9)
res.fit(X)

print(res.n_components_)

res = PCA(n_components=10)
now = res.fit_transform(X)[:,0]

index = pandas.read_csv('djia_index.csv')
print(np.corrcoef(now, index['^DJI'])[1][0])

print(max(list(zip(now, pandas.read_csv('close_prices.csv').columns[1:])))[1])
