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

selected_models = ['ST8000DM002', 'ST4000DX000', 'ST12000NM0007', 'ST4000DM000','ST8000NM0055','ST3000DM001']
# selected_models = ['ST4000DM000']
out_file = open("../lookup_window_perf/result.txt","a+")
for model in selected_models:
    for day in range(1,8):
        print(model," ",day)
        
        hdd = pd.read_csv("../final_dataset/"+str(day)+'/'+str(model)+'.csv', header=None)
        #hdd = pd.read_csv("../dataset_1.csv")
        hdd = hdd.drop(hdd.columns[6], axis=1)
        hdd = hdd.drop(hdd.columns[9], axis=1)
        hdd = hdd.drop(hdd.columns[14], axis=1)
        hdd = hdd.drop(hdd.columns[13], axis=1)
        hdd = hdd.dropna()

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

        # train_dataset = train_dataset.dropna()

        x_new = train_dataset.iloc[:, :-1].values
        y_new = train_dataset.iloc[:, -1].values

        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler()

        X_resampled, y_resampled = rus.fit_resample(x_new, y_new)
        print(Counter(y_resampled))

        clf= RandomForestClassifier()
        clf.fit(X_resampled,y_resampled)

        # test_dataset = test_dataset.dropna()
                    
        X_test = test_dataset.iloc[:, :-1].values
        y_test = test_dataset.iloc[:, -1].values
        y_pred=clf.predict(X_test)
        preds = clf.predict_proba(X_test)
        print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
        print("Recall: ",metrics.recall_score(y_test, y_pred))
        conf_matrix = metrics.confusion_matrix(y_test, y_pred)
        print(conf_matrix)
        print(metrics.roc_auc_score(y_test, y_pred))
        TP, FP, TN, FN = perf_measure(y_test, y_pred)

        recall = (TP/(TP+FN))*100
        fpr = (FP/(TN+FP))*100

        # for index in range(len(y_test)):
        out_file.write(model+"\t"+str(day)+"\t"+str(recall)+"\t"+str(fpr)+"\n")
        out_file.flush()
        # out_file.close()