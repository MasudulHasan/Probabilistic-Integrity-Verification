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

selected_models = ['ST8000DM002', 'ST12000NM0007', 'ST4000DM000','ST8000NM0055']
# selected_models = ['ST4000DM000']
# for iter_ in range(5):
for model in selected_models:
    print(model)
    # out_file = open("../model_perf_result_with_old_data/"+str(model)+"_"+str(iter_+1)+".txt","a+")
    # df_list = []
    # for year in range(2013,2019):
    #     try:
    #         hdd = pd.read_csv("../dataset_"+str(year)+"/1/"+str(model)+'.csv', header=None)
    #         hdd = hdd.drop(hdd.columns[17], axis=1)
    #         hdd = hdd.drop(hdd.columns[16], axis=1)
    #         hdd = hdd.drop(hdd.columns[11], axis=1)
    #         hdd = hdd.drop(hdd.columns[7], axis=1)
    #         hdd = hdd.drop(hdd.columns[0], axis=1)
    #         hdd = hdd.dropna()
    #         df_list.append(hdd_extra)
    #     except:
    #         pass

    hdd1 = pd.read_csv("../final_dataset/"+str(1)+'/'+str(model)+'.csv', header=None)
    #hdd = pd.read_csv("../dataset_1.csv")
    # hdd1 = hdd1.drop(hdd1.columns[6], axis=1)
    hdd1 = hdd1.drop(hdd1.columns[10], axis=1)
    hdd1 = hdd1.drop(hdd1.columns[15], axis=1)
    hdd1 = hdd1.drop(hdd1.columns[14], axis=1)
    hdd1 = hdd1.dropna()
    # print(hdd.head())
    month_to_be_considerate = 8

    while(month_to_be_considerate <= 9):
        prev_month = 1
        df_list = []
        year_to_cosider=2019
        while(prev_month < month_to_be_considerate):
            hdd_extra = pd.read_csv("../month_wise_file/"+str(year_to_cosider)+"/"+str(prev_month)+'/'+str(model)+'.csv', header=None)
            hdd_extra = hdd_extra.drop(hdd_extra.columns[10], axis=1)
            hdd_extra = hdd_extra.dropna()
            df_list.append(hdd_extra)
            prev_month += 1

        df_list.append(hdd1)
        result = pd.concat(df_list)

        x = result.iloc[:, :-1].values
        y = result.iloc[:, -1].values

        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler()

        X_resampled, y_resampled = rus.fit_resample(x, y)
        print(Counter(y_resampled))
        clf = RandomForestClassifier()
        clf.fit(X_resampled, y_resampled)

        hdd_test = pd.read_csv("../month_wise_file/"+str(year_to_cosider)+"/"+str(month_to_be_considerate)+'/'+str(model)+'.csv', header=None)
        hdd_test = hdd_test.drop(hdd_test.columns[10], axis=1)
        hdd_test = hdd_test.dropna()

        X_test = hdd_test.iloc[:, :-1].values
        y_test = hdd_test.iloc[:, -1].values
        y_pred=clf.predict(X_test)
        preds = clf.predict_proba(X_test)
        print("month ", month_to_be_considerate)
        print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
        print("Recall: ",metrics.recall_score(y_test, y_pred))
        conf_matrix = metrics.confusion_matrix(y_test, y_pred)
        print(conf_matrix)
        print(metrics.roc_auc_score(y_test, y_pred))
        # print(perf_measure(y_test, y_pred))
        # out_file = open("../training_window_test_result/"+disk_model_name+"_"+str(month_to_be_considerate)+"_"+str(window)+".txt","a+")
        # for index in range(len(y_test)):
        #     out_file.write(str(y_test[index])+"\t"+str(y_pred[index])+"\t"+str(preds[index])+"\n")
        #     out_file.flush()
        # out_file.close()

        month_to_be_considerate+=1