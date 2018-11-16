from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from image_process_cs2 import data_preprocess
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.naive_bayes import GaussianNB

import numpy as np

components, y = data_preprocess(['japanese_beetle', 'cucumber_beetle', 'ladybug'], (200,200,3), True)

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(components) # standardize data

pca = PCA(n_components=120) #pca object
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, stratify=y, random_state= 42)

bac = make_scorer(balanced_accuracy_score)

clf = RandomForestClassifier()
parameters = {
    'n_estimators':[100, 500, 1000],
    'max_features':['auto', 'log2'],
    'min_samples_leaf': [2, 4, 5, 8],
    'min_impurity_decrease':[0, 0.01, 0.05, 0.1, 0.2]
    }
grid = GridSearchCV(clf, parameters, scoring=bac, cv=5)
grid.fit(X_train, y_train)
print(grid.best_score_)
print(grid.best_estimator_)
grid_score = grid.score(X_test, y_test)
print(grid_score)

# 0.4618736383442267
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=None, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.01, min_impurity_split=None,
#             min_samples_leaf=2, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=None,
#             oob_score=False, random_state=None, verbose=0,
#             warm_start=False)
# 0.5666666666666668

ada = AdaBoostClassifier()
ada_params = {
    'n_estimators': [50, 100, 500, 1000],
    'learning_rate': [0.1, 0.5, 1, 5, 10 ]
    }
ada_grid = GridSearchCV(ada, ada_params, scoring=bac, cv=5)
ada_grid.fit(X_train, y_train)
print(ada_grid.best_score_)
print(ada_grid.best_estimator_)
ada_grid_score = ada_grid.score(X_test, y_test)
print(ada_grid_score)

# 0.4483788827298383
# AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
#           n_estimators=100, random_state=None)
# 0.45

nb = GaussianNB()
nb_params = {
'var_smoothing':[1e-10, 1e-9, 1e-8, 1e-7]
}
nb_grid = GridSearchCV(nb, nb_params, scoring=bac, cv=5)
nb_grid.fit(X_train, y_train)
print(nb_grid.best_score_)
print(nb_grid.best_estimator_)
nb_grid_score = nb_grid.score(X_test, y_test)
print(nb_grid_score)

# 0.4170972570079048
# GaussianNB(priors=None, var_smoothing=1e-10)
# 0.4333333333333333
