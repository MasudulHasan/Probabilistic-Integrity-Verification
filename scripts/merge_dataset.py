#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 05:31:15 2019

@author: masudulhasanmasudb
"""
import time
import glob,random
import datetime
import os
import subprocess
import shlex
import gc
import pandas as pd
import collections

#def perf_measure(y_actual, y_hat):
#    TP = 0
#    FP = 0
#    TN = 0
#    FN = 0
#
#    for i in range(len(y_hat)): 
#        if y_actual[i]==y_hat[i]==1:
#           TP += 1
#        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
#           FP += 1
#        if y_actual[i]==y_hat[i]==0:
#           TN += 1
#        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
#           FN += 1
#    
#    print(TP/(TP+FN)) 
#    
#    print(FP/(FP+TN))       
#           
#    return(TP, FP, TN, FN)
#
#def count_unique(keys):
#    uniq_keys = np.unique(keys)
#    bins = uniq_keys.searchsorted(keys)
#    return uniq_keys, np.bincount(bins)

parent_folder_name = "data_files/"
folder_list=glob.glob(parent_folder_name+"*")
#for x in folder_list:
#    print(x)
#
#print(folder_list)

#output_file = open("dataset_1.csv","a+")
#
#for x in folder_list:
#    with open(x,"r") as in_file:
#        for line in in_file:
##            print(line)
#            if "label" not in line:
#                output_file.write(line)
#                output_file.flush()
#
count =0 
p_count = 0
with open("../dataset_1.csv","r") as in_file:
        for line in in_file:
#            print(line)
            count+=1
            parts = line.split(",")
            if(int(parts[len(parts)-1])==1):
                p_count+=1

print(p_count)
print(count)
print((p_count/count)*100) 

#import numpy as np
#import pandas as pd
##import seaborn as sns
#import matplotlib.pyplot as plt
#from sklearn import ensemble, metrics 
#import gc
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
#from sklearn import metrics
#from imblearn.over_sampling import (RandomOverSampler, SMOTE, ADASYN)
#from collections import Counter
#
#hdd = pd.read_csv('dataset_1.csv')
##hdd = hdd.dropna()
##print(hdd.iloc[0]) 
#hdd = hdd.drop(hdd.columns[6], axis=1)
#hdd = hdd.drop(hdd.columns[9], axis=1)
##print(hdd.iloc[0])
#
#hdd = hdd.dropna()
#
#hdd_pos = hdd.loc[hdd[hdd.columns[12]] == 1]
##print(len(hdd_pos))
#
##hdd_pos_over = hdd_pos.sample(len(hdd_pos), replace=True)
#
##hdd = hdd.drop(hdd[hdd.columns[12]] == 1, axis=0)
##print(hdd.head)
#
#hdd = hdd[hdd[hdd.columns[12]] == 0]
##print(hdd.head)
#
#print(len(hdd))
#
##hdd_neg = hdd.sample(n = len(hdd_pos), random_state = 2) 
#hdd_neg = hdd
#
#
#
##print(len(hdd_neg))
##print(hdd.columns[12], axis=1)
#
#hdd_merged = [hdd_pos, hdd_neg]
#
#result = pd.concat(hdd_merged)
##print(result.head)
#
#x = result.iloc[:, :-1].values
#y = result.iloc[:, -1].values
#
#X_resampled, y_resampled = SMOTE().fit_resample(x, y)
#print(sorted(Counter(y_resampled).items()))
#print(len(y_resampled))
#
#
##max_th = dataset.iloc[:, 13:14].values
#
##print(x[:10])
##print(y[:10])
##print(len(x))
##print(type(x))
#
#
#
#
##X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
#
#X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)
#
#print(count_unique(y_train))
#print(count_unique(y_test))
##
##real_x = dataset.iloc[:, :].values
#
##from sklearn.ensemble import RandomForestClassifier
##
###Create a Gaussian Classifier
#clf=RandomForestClassifier()
#
##regr = RandomForestRegressor(max_depth=2, random_state=0,
##                              n_estimators=100)
#
##X_train = X_train.fillna(X_train.mean())
#
##print(np.where(np.isnan(X_train)))
##print(type(X_train))
##print(np.where(np.isnan(y_train)))
#
##Train the model using the training sets y_pred=clf.predict(X_test)
#clf.fit(X_train,y_train)
#y_pred=clf.predict(X_test)
#
##preds = clf.predict_proba(X_test)
##out_file = open("output.txt","a+")
##for x in preds:
##    out_file.write(str(x)+"\n")
##    out_file.flush()
##print(preds[:,1])
##print(collections.Counter(preds[:,1]))
##print(preds.shape)
##print(preds)
##print(preds[:,1])
##print(test_labels)
#
#print(y_pred)
#print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
#print("Recall: ",metrics.recall_score(y_test, y_pred))
#conf_matrix = metrics.confusion_matrix(y_test, y_pred)
#print(conf_matrix)
#print(perf_measure(y_test, y_pred))              