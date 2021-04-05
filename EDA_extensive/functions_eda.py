# -*- coding: utf-8 -*-
"""functions_eda

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1A4DIZIiPIUXAjDPIQxT80puZGcOUcBuQ
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn import metrics
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from scipy import stats

"""### **Summary Statistics functions:**

##### **Finding Missing data:**
"""

def missing_data_finder(df):

    df_missing = df.isnull().sum().reset_index().rename(columns={'index': 'column_name', 0: 'missing_row_count'}).copy()
    df_missing_rows = df_missing[df_missing['missing_row_count'] > 0].sort_values(by='missing_row_count',ascending=False)
    df_missing_rows['missing_row_percent'] = (df_missing_rows['missing_row_count'] / df.shape[0]).round(4)
    return df_missing_rows

"""##### **Printing Full Summary for data:**"""

def full_summary_table(df):

    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary

"""##### **Calculating Outliers:**"""

def calculate_outliers(df_num): 

    # calculating mean and std of the array
    data_mean, data_std = np.mean(df_num), np.std(df_num)
    cut = data_std * 3

    #Calculating the higher and lower cut values
    lower, upper = data_mean - cut, data_mean + cut

    # creating an array of lower, higher and total outlier values 
    outliers_lower = [x for x in df_num if x < lower]
    outliers_higher = [x for x in df_num if x > upper]
    outliers_total = [x for x in df_num if x < lower or x > upper]

    # array without outlier values
    outliers_removed = [x for x in df_num if x > lower and x < upper]
    
    print('Identified lowest outliers: %d' % len(outliers_lower)) # printing total number of values in lower cut of outliers
    print('Identified upper outliers: %d' % len(outliers_higher)) # printing total number of values in higher cut of outliers
    print('Total outlier observations: %d' % len(outliers_total)) # printing total number of values outliers of both sides
    print('Non-outlier observations: %d' % len(outliers_removed)) # printing total number of non outlier values
    print("Total percentual of Outliers: ", round((len(outliers_total) / len(outliers_removed) )*100, 4)) # Percentual of outliers in points
    
    return

"""### **Plotting Functions:**

##### **Plotting distribution ratio:**
"""

def plotting_dist_ratio(df, col, lim=2000):

    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    plt.figure(figsize=(15,5))
    plt.suptitle(f'{col} Distributions ', fontsize=15)

    plt.subplot(121)
    g = sns.countplot(x=col, data=df, order=list(tmp[col].values))
    g.set_title(f"{col} Distribution\nCount and Fraud(%) by each category")
    g.set_ylim(0,400000)
    gt = g.twinx()
    gt = sns.pointplot(x=col, y='Fraud', data=tmp, order=list(tmp[col].values),
                       color='black', legend=False, )
    gt.set_ylim(0,20)
    gt.set_ylabel("% of Fraud Transactions")
    g.set_xlabel(f"{col} Category Names")
    g.set_ylabel("Count")
    for p in gt.patches:
        height = p.get_height()
        gt.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/len(df)*100),
                ha="center",fontsize=14) 
        
    perc_amt = (df.groupby(['isFraud',col])['TransactionAmt'].sum() / (df.groupby([col])['TransactionAmt'].sum()* 100)).unstack('isFraud')
    perc_amt = perc_amt.reset_index()
    perc_amt.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    plt.subplot(122)
    g1 = sns.boxplot(x=col, y='TransactionAmt', hue='isFraud', 
                     data=df[df['TransactionAmt'] <= lim], order=list(tmp[col].values))
    g1t = g1.twinx()
    g1t = sns.pointplot(x=col, y='Fraud', data=perc_amt, order=list(tmp[col].values),
                       color='black', legend=False, )
    g1t.set_ylim(0,5)
    g1t.set_ylabel("\nTotal Fraud (%)")
    g1.set_title(f"{col} by Distribution of TransactionAmt ")
    g1.set_xlabel(f"{col} Category Names")
    g1.set_ylabel("Transaction Amount")
        
    plt.subplots_adjust(hspace=.4, wspace = 0.35, top = 0.80)
    
    plt.show()

"""##### **Plotting Count Amounts:**"""

def plotting_cnt_amt(df, col, lim=2000):
    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)
    
    #Distribution plot
    plt.figure(figsize=(15,8))    
    plt.suptitle(f'{col} Distributions ', fontsize=20)
    
    plt.subplot(211)
    g = sns.countplot( x=col,  data=df, order=list(tmp[col].values))
    gt = g.twinx()
    gt = sns.pointplot(x=col, y='Fraud', data=tmp, order=list(tmp[col].values),
                       color='black', legend=False, )
    gt.set_ylim(0,tmp['Fraud'].max()*1.1)
    gt.set_ylabel("\nFraud(%) Transactions")
    g.set_title(f"Distribution of {col} and corresponding Fraud %", fontsize=15)
    g.set_xlabel(f"{col} Category Names")
    g.set_ylabel("Count")
    g.set_xticklabels(g.get_xticklabels(),rotation=45)
    
    plt.show()

"""##### **Plotting Categorical Features:**"""

def plotting_cat_feat(df, col):

    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    plt.figure(figsize=(15,10))
    plt.suptitle(f'{col} Distributions',fontsize=20)

    plt.subplot(221)
    g = sns.countplot(x=col, data=df, order=tmp[col].values)

    g.set_title(f"{col} Distribution\n")
    g.set_xlabel(f"{col} Name")
    g.set_ylabel("Count")

    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/(len(df))*100),
                ha="center", fontsize=14) 

    plt.subplot(222)
    g1 = sns.countplot(x=col, hue='isFraud', data=df, order=tmp[col].values)
    plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])
    gt = g1.twinx()
    gt = sns.pointplot(x=col, y='Fraud', data=tmp, color='black', order=tmp[col].values, legend=False)
    gt.set_ylabel("\n% of Fraud Transactions")

    g1.set_title(f"{col} by Target(isFraud)\n")
    g1.set_xlabel(f"{col} Name")
    g1.set_ylabel("Count")

    plt.subplot(212)
    g3 = sns.boxenplot(x=col, y='TransactionAmt', hue='isFraud', 
                       data=df[df['TransactionAmt'] <= 2000], order=tmp[col].values )
    g3.set_title("Transaction Amount Distribuition by ProductCD and Target")
    g3.set_xlabel("ProductCD Name\n")
    g3.set_ylabel("Transaction Values")

    plt.subplots_adjust(hspace = 0.4, top = 0.85)

    plt.show()