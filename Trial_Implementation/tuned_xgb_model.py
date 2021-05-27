import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sys import argv



"""### **Importing the data - (Here we are using the resampled data from our Class_Imbalance notebook)**"""

X_train = pd.read_csv('/Users/akhil/Desktop/Capstone:ExtraProject/ieee-fraud-detection/Class Imbalance/resampled data/X_train.csv')
y_train = pd.read_csv('/Users/akhil/Desktop/Capstone:ExtraProject/ieee-fraud-detection/Class Imbalance/resampled data/y_train.csv')
X_test = pd.read_csv('/Users/akhil/Desktop/Capstone:ExtraProject/ieee-fraud-detection/Class Imbalance/resampled data/X_test.csv')


def data_manip():

    """**Filling NaN values notarized by -1 to -999**"""
    X_train.replace(to_replace = -1, value = -999, inplace=True)
    y_train.replace(to_replace = -1, value = -999, inplace=True)
    X_test.replace(to_replace = -1, value = -999, inplace=True)

    X_train.drop('Unnamed: 0',axis=1, inplace=True)
    y_train.drop('Unnamed: 0',axis=1, inplace=True)
    X_test.drop('Unnamed: 0',axis=1, inplace=True)

    print('X_train shape: {}'.format(X_train.shape))
    print('y_train shape: {}'.format(y_train.shape))
    print('X_test shape: {}'.format(X_test.shape))


data_manip()
y_train = y_train['isFraud'].astype(bool)

import memory_reduction
from memory_reduction import reduce_mem_usage

"""**Reducing memory usage for datasets**"""

X_train = reduce_mem_usage(X_train)
y_train = reduce_mem_usage(y_train)
X_test = reduce_mem_usage(X_test)

"""## **2. XGBoost Classifier (tuned params)**
* ### **Parameter Tuning with Stratified-KFold & RandomizedSearchCV**
"""

import xgboost
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

"""**Splitting training data into training and validation sets**"""

x_tra, x_val, y_tra, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42 )

def param_search():

    # Create another default XGBoost classifier
    model = xgboost.XGBClassifier(
        eval_set=[(x_tra, y_tra),(x_val,y_val)],
        eval_metric=['auc','error'],
        random_state = 42
        )

    # Create the grid search parameter grid and scoring funcitons
    param_grid = {
        "learning_rate": [0.01,0.1,0.4,0.8],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "subsample": [0.6, 0.8, 1.0],
        "max_depth": [3, 9, 15, 21],
        "n_estimators": [100, 250, 500],
        "reg_lambda": [1, 1.5, 2],
        "min_child_weight": [0, 0.1, 0.3],
        }

    scoring = {
        'AUC': 'roc_auc',
        'Accuracy': make_scorer(accuracy_score)
        }

    # Create the Kfold object
    num_folds = 5
    kfold = StratifiedKFold(n_splits=num_folds)

    # Create the grid search object
    n_iter=50
    grid = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        cv=kfold,
        scoring=scoring,
        n_iter=n_iter,
        refit="AUC",
        )

    # Fit the grid search
    best_model = grid.fit(x_tra,y_tra)

    # Print the best parameter results
    print(f'Best score: {best_model.best_score_}')
    print(f'Best model: {best_model.best_params_}')

param_search()
