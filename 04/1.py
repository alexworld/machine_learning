import numpy as np
import pandas
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

data_train = pandas.read_csv('salary-train.csv')
data_test = pandas.read_csv('salary-test-mini.csv')

for i in range(len(data_train)):
    for j in ['FullDescription']:
        data_train.set_value(i, j, data_train.get_value(i, j).lower())

for i in range(len(data_test)):
    for j in ['FullDescription']:
        data_test.set_value(i, j, data_test.get_value(i, j).lower())
        
data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)

vect = TfidfVectorizer(min_df=5)
X_train_text = vect.fit_transform(raw_documents=data_train['FullDescription'])
X_test_text = vect.transform(raw_documents=data_test['FullDescription'])

data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)

data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_train = hstack([X_train_text, X_train_categ])
X_test = hstack([X_test_text, X_test_categ])

y_train = np.ravel(data_train['SalaryNormalized'].as_matrix())

res = Ridge(alpha=1)
res.fit(X_train, y_train)

print(res.predict(X_test))
