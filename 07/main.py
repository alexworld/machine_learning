import numpy as np
import pandas
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime

data_train_all = pandas.read_csv('features.csv', index_col='match_id')
data_test = pandas.read_csv('features_test.csv', index_col='match_id')
data_train = data_train_all.drop(list(set(data_train_all.columns) - set(data_test.columns)), axis=1)

count_columns = np.ravel(data_train.count().as_matrix())
max_empty_rate = 0;

print('Next columns have empty values:')

for i in range(len(count_columns)):
    if count_columns[i] < len(data_train[[i]]):
        print(data_train.columns[i], end = ' ')
    max_empty_rate = max(max_empty_rate, (len(data_train[[i]]) - count_columns[i]) / len(data_train[[i]]))
print()
print()
print('Maximum empty rate in one column:', max_empty_rate)
print()

for col in data_train.columns:
    data_train[col].fillna(0, inplace=True)

target_column = [col for col in (set(data_train_all.columns) - set(data_test.columns)) if 'status' not in col and col != 'duration'][0]
print('Target column:', target_column)
print()

X_train = data_train.as_matrix()
y_train = np.ravel(data_train_all[target_column])

print('Gradient Boosting:')

for n_estimators in [10, 20, 30, 40]:
    start_time = datetime.now()
    clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=241)
    clf.fit(X_train, y_train)

    val = KFold(n_folds=5, shuffle=True, random_state=241)
    print('For n_estimators =', n_estimators, 'the validation score is', cross_val_score(X=X_train, y=y_train, estimator=clf, cv=val).mean())
    print('Time elapsed:', datetime.now() - start_time)
    print()
print()


def check(X_train, y_train):
    best_C = 1
    best_score = 0
    start_time = datetime.now()

    for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        clf = LogisticRegression(penalty='l2', C=C, random_state=241)
        clf.fit(X_train, y_train)
    
        val = KFold(n_folds=5, shuffle=True, random_state=241)
        score = cross_val_score(X=X_train, y=y_train, estimator=clf, cv=val, scoring='roc_auc').mean()

        if score > best_score:
            bestC, best_score = C, score

    print('Best C =', best_C)
    print('Best score =', best_score)
    print('Average time elapsed:', (datetime.now() - start_time) / 7)
    print()
    return best_C

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

print('Logistic Regression:')
check(X_train, y_train)

print('Removing category features')

data_train_copy = data_train.copy(deep=True)
data_train.drop('lobby_type', axis=1, inplace=True)

for i in range(1, 6):
    data_train.drop('r' + str(i) + '_hero', axis=1, inplace=True)
    data_train.drop('d' + str(i) + '_hero', axis=1, inplace=True)

X_train = data_train.as_matrix()
X_train = scaler.fit_transform(X_train)
check(X_train, y_train)

data_train = data_train_copy

hero_vals = []

for i in range(1, 6):
    hero_vals += data_train['r' + str(i) + '_hero'].tolist()
    hero_vals += data_train['d' + str(i) + '_hero'].tolist()

hero_vals = sorted(list(set(hero_vals)))
N = len(hero_vals)
print('Number of heroes is', N)
print()
print('Transforming features')


def transform_features(data):
    X_pick = np.zeros((data.shape[0], N))

    for i, match_id in enumerate(data.index):
        for p in range(5):
            X_pick[i, hero_vals.index(data.ix[match_id, 'r%d_hero' % (p + 1)])] = 1
            X_pick[i, hero_vals.index(data.ix[match_id, 'd%d_hero' % (p + 1)])] = -1

    X = np.hstack((data.as_matrix(), X_pick))
    return X

X_train = transform_features(data_train)
X_train = scaler.fit_transform(X_train)
C = check(X_train, y_train)

clf = LogisticRegression(penalty='l2', C=C, random_state=241)
clf.fit(X_train, y_train)

for col in data_test.columns:
    data_test[col].fillna(0, inplace=True)
X_test = transform_features(data_test)
X_test = scaler.transform(X_test)
pred = clf.predict_proba(X_test)[:, 1]

out = [[data_test.index.values[i], pred[i]] for i in range(len(pred))]
f = open('output.csv', 'w')
f.write('match_id,radiant_win\n')

for x in out:
    f.write(str(x[0]) + ',' + str(x[1]) + '\n')

print('Minimal prediction:', min(list(pred)))
print('Maximal prediction:', max(list(pred)))
