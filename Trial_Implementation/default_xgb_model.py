import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sys import argv



"""### **Importing the data - (Here we are using the resampled data from our Class_Imbalance notebook)**"""

X_train = pd.read_csv('/Users/akhil/Desktop/Capstone:ExtraProject/ieee-fraud-detection/Trial_Models/data/X_train.csv')
y_train = pd.read_csv('/Users/akhil/Desktop/Capstone:ExtraProject/ieee-fraud-detection/Trial_Models/data/y_train.csv')
X_test = pd.read_csv('/Users/akhil/Desktop/Capstone:ExtraProject/ieee-fraud-detection/Trial_Models/data/X_test.csv')


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

#"""**Reducing memory usage for datasets**"""

#X_train = reduce_mem_usage(X_train)
#y_train = reduce_mem_usage(y_train)
#X_test = reduce_mem_usage(X_test)


"""## **1. XGBoost - Basic Classifier**
* ### **Default parameters with eval_set and eval_metric**
"""

import xgboost

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def xgboost_default():
    """**Splitting training data into training and validation sets**"""

    x_tra, x_val, y_tra, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)


    """* #### **Implementing and training a default XGBoost classifier**"""

    xgb = xgboost.XGBClassifier(n_estimators=250, random_state=200, seed=42, use_label_encoder = False)

    # define the eval set and metric
    eval_set=[(x_tra, y_tra),(x_val,y_val)]
    eval_metric = ["auc", "error"]

    # fit the model
    xgb.fit(x_tra, y_tra, eval_set = eval_set, eval_metric=eval_metric, verbose=False)

    """**Predicting target variable for x_val and classification report**"""

    y_pred_val = xgb.predict(x_val)
    y_pred_proba = xgb.predict_proba(x_val)

    # Model assessment
    print(classification_report(y_val, y_pred_val))
    print('\nAccuracy on Validation set: {: .3f}'.format(accuracy_score(y_val, y_pred_val)))
    print('Error: {: .3f}'.format(1 - accuracy_score(y_val, y_pred_val)))

    """* #### **Plotting performance eval metrics (make into .py function)**"""

    # retrieve and plot performance metrics
    results = xgb.evals_result_
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)

    fig, ax = plt.subplots(1, 2, figsize=(18,6))

    # plot auc-roc curve
    ax[0].plot(x_axis, results['validation_0']['auc'], label='Train')
    ax[0].plot(x_axis, results['validation_1']['auc'], label='Val')
    ax[0].legend(fontsize=13)
    ax[0].set_title('XGBoost AUC-ROC\n', fontsize=15)
    ax[0].set_ylabel('AUC-ROC\n')
    ax[0].set_xlabel('\nN estimators')

    # plot classification error
    ax[1].plot(x_axis, results['validation_0']['error'], label='Train')
    ax[1].plot(x_axis, results['validation_1']['error'], label='Val')
    ax[1].legend(fontsize=13)
    ax[1].set_title('XGBoost Classification Error\n', fontsize=15)
    ax[1].set_ylabel('Classification Error\n')
    ax[1].set_xlabel('\nN estimators')

    plt.show()
    plt.tight_layout()

    """* ### **Feature Importance Visualization (make into .py function)**"""

    type(xgb.feature_importances_)

    plot_df = pd.DataFrame([np.array(list(X_train.columns)),
                            list(xgb.feature_importances_)]).T
    plot_df.columns = ['Feature Name', 'Score']
    plot_df.sort_values('Score', ascending = False, inplace = True)
    plot_df.set_index('Feature Name', inplace = True)

    plot_df.iloc[:55].plot(kind = 'bar', legend = False, figsize = (16,7),
                            color="g", align="center")
    plt.title("Feature Importance\n", fontsize=20)
    plt.xticks(fontsize = 10)
    plt.xlabel('Feature Names', fontsize=18)
    plt.ylabel('Score\n', fontsize =18)
    plt.show()

xgboost_default()

# Dropping the columns with a score of 0
X_train = X_train[['C8','C5','R_emaildomain_bin', 'PCA_V_5','C12','C2','PCA_V_14','C11','D3','C14',
                    'M4','C1','card6','C7','C4','D5','D2','PCA_V_7','PCA_V_2','TransactionAmt',
                    'PCA_V_19','C6','PCA_V_9', 'C2_to_mean_addr2','D15','M5','PCA_V_6','C1_to_std_addr2',
                    'D1_to_std_card1', 'M6','TransactionID','D8','C13','C2_to_mean_addr1','card5',
                    'PCA_V_8','card1','D11','PCA_V_27','PCA_V_16','C2_to_std_addr2','D1_to_mean_card4',
                    'D6','D4','PCA_V_10','M3','D10_to_mean_card5','C9','card2','C2_to_std_addr1','PCA_V_25',
                    'C2_to_std_dist1','D1_to_mean_card5','dist1','C1_to_std_addr1', 'TransactionAmt_to_std_card5',
                    'PCA_V_13','D1_to_std_card4','PCA_V_3','PCA_V_17','PCA_V_0','addr1','PCA_V_11','P_emaildomain_bin',
                    'PCA_V_22', 'C10','PCA_V_15','D1_to_std_card5','PCA_V_18','D10_to_mean_card1','C2_to_mean_dist1',
                    'TransactionAmt_to_std_card1','D1_to_mean_card1','PCA_V_1', 'C1_to_mean_addr1','P_emaildomain_suffix',
                    'D14','PCA_V_26','TransactionAmt_to_mean_card5','D1','PCA_V_24','C3','PCA_V_29','TransactionAmt_to_mean_card1',
                    'C1_to_std_dist1','TransactionAmt_to_mean_card4', 'PCA_V_4','PCA_V_20','M9','M2','PCA_V_23',
                    'V2','D10_to_std_card5','R_emaildomain_suffix','id_31','id_17','PCA_V_12','id_02_to_mean_card1',
                    'id_34','D9','D10_to_mean_card4']]

X_test = X_test[['C8','C5','R_emaildomain_bin', 'PCA_V_5','C12','C2','PCA_V_14','C11','D3','C14',
                    'M4','C1','card6','C7','C4','D5','D2','PCA_V_7','PCA_V_2','TransactionAmt',
                    'PCA_V_19','C6','PCA_V_9', 'C2_to_mean_addr2','D15','M5','PCA_V_6','C1_to_std_addr2',
                    'D1_to_std_card1', 'M6','TransactionID','D8','C13','C2_to_mean_addr1','card5',
                    'PCA_V_8','card1','D11','PCA_V_27','PCA_V_16','C2_to_std_addr2','D1_to_mean_card4',
                    'D6','D4','PCA_V_10','M3','D10_to_mean_card5','C9','card2','C2_to_std_addr1','PCA_V_25',
                    'C2_to_std_dist1','D1_to_mean_card5','dist1','C1_to_std_addr1', 'TransactionAmt_to_std_card5',
                    'PCA_V_13','D1_to_std_card4','PCA_V_3','PCA_V_17','PCA_V_0','addr1','PCA_V_11','P_emaildomain_bin',
                    'PCA_V_22', 'C10','PCA_V_15','D1_to_std_card5','PCA_V_18','D10_to_mean_card1','C2_to_mean_dist1',
                    'TransactionAmt_to_std_card1','D1_to_mean_card1','PCA_V_1', 'C1_to_mean_addr1','P_emaildomain_suffix',
                    'D14','PCA_V_26','TransactionAmt_to_mean_card5','D1','PCA_V_24','C3','PCA_V_29','TransactionAmt_to_mean_card1',
                    'C1_to_std_dist1','TransactionAmt_to_mean_card4', 'PCA_V_4','PCA_V_20','M9','M2','PCA_V_23',
                    'V2','D10_to_std_card5','R_emaildomain_suffix','id_31','id_17','PCA_V_12','id_02_to_mean_card1',
                    'id_34','D9','D10_to_mean_card4']]


def save_data():

    X_train.to_csv('X_train_final.csv')
    y_train.to_csv('y_train_final.csv')
    X_test.to_csv('X_test_final.csv')

save_data()
