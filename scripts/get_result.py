import time
import glob,random
import datetime
import os
import subprocess
import shlex
import gc,sys, traceback
import pandas as pd
import collections
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import ensemble, metrics 
import gc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from imblearn.over_sampling import (RandomOverSampler, SMOTE, ADASYN)
from collections import Counter

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    
    print(TP/(TP+FN)) 
    
    print(FP/(FP+TN))       
           
    return(TP, FP, TN, FN)
    
def calculate_accuracy(y_pred, y_test, pred_list):
    output_str=""
    for x in range(len(y_pred)):
        output_str+=str(y_pred[x])+" "+str(y_test[x])+" "+str(pred_list[x])+"\n"
    
    return output_str
out_file = open("result.txt","a+")
hdd = pd.read_csv("/data/masud/final_dataset_corrected/ST4000DM000.csv", header=None)
hdd = hdd.drop(hdd.columns[17], axis=1)
hdd = hdd.drop(hdd.columns[16], axis=1)
hdd = hdd.drop(hdd.columns[11], axis=1)
hdd = hdd.drop(hdd.columns[7], axis=1)
hdd = hdd.drop(hdd.columns[0], axis=1)
hdd = hdd.dropna()
hdd = hdd.T.reset_index(drop=True).T
hdd_pos = hdd.loc[hdd[hdd.columns[12]] == 1]
hdd_neg = hdd[hdd[hdd.columns[12]] == 0]
del hdd
gc.collect()

pos_train, pos_test = train_test_split(hdd_pos, test_size=0.2)
neg_train, neg_test = train_test_split(hdd_neg, test_size=0.2)

train_merged = [pos_train, neg_train]
train_dataset = pd.concat(train_merged)

test_merged = [pos_test, neg_test]
test_dataset = pd.concat(test_merged)

train_dataset = train_dataset.dropna()

x_new = train_dataset.iloc[:, :-1].values
y_new = train_dataset.iloc[:, -1].values

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler()

X_resampled, y_resampled = rus.fit_resample(x_new, y_new)
print(Counter(y_resampled))

clf= RandomForestClassifier()
clf.fit(X_resampled,y_resampled)

test_dataset = test_dataset.dropna()
            
X_test = test_dataset.iloc[:, :-1].values
y_test = test_dataset.iloc[:, -1].values
y_pred=clf.predict(X_test)
preds = clf.predict_proba(X_test)

prob_score = calculate_accuracy(y_pred,y_test,preds)

print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
print("Recall: ",metrics.recall_score(y_test, y_pred))

# out_file.write("\n\n"+"model: "+model_name+"\n")
# out_file.write("\n")
out_file.write("Accuracy: "+ str(metrics.accuracy_score(y_test, y_pred))+"\n")
out_file.write("Recall: "+ str(metrics.recall_score(y_test, y_pred))+"\n")
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
out_file.write(str(conf_matrix))
out_file.write("\n")
out_file.write(str(perf_measure(y_test, y_pred)))
out_file.write("\n")
out_file.write(prob_score)
out_file.flush()

print(metrics.roc_auc_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))