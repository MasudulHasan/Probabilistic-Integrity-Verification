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
import gc, traceback
import pandas as pd
import collections

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

def count_unique(keys):
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return uniq_keys, np.bincount(bins)


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

parent_folder_name = "/home/masudulhasanmasudb/Music/hdd_data/dataset_2017/"
folder_list=glob.glob(parent_folder_name+"*")
#print(folder_list)

def all_same(items):
    return all(x == items[0] for x in items)

count=0
for folder in folder_list:
    file_list=glob.glob(folder+"/*")
#    print(file_list)
    start_index = folder.find("_2017")
#    end_index = file.rfind("/")
    day_number = folder[start_index+6:]
    print(day_number)
    out_file = open("/home/masudulhasanmasudb/Music/hdd_data/result_2017/"+day_number+"/result_bayesian.txt","a+")
    for file in file_list:
#        print(file)
        start_index = file.rfind("/")
        end_index = file.rfind(".")
        model_name = file[start_index+1:end_index]        
        try:
            hdd = pd.read_csv(file)
            hdd = hdd.drop(hdd.columns[6], axis=1)
            hdd = hdd.drop(hdd.columns[9], axis=1)
    #        print(hdd.head)
            hdd = hdd.dropna(axis='columns',how='all')
            hdd = hdd.dropna()
    #        print(hdd.head)
            
            x = hdd.iloc[:, :-1].values
            y = hdd.iloc[:, -1].values
#            print(sorted(Counter(y).items()))
            if not all_same(y):
#                print(file)
#                print(sorted(Counter(y).items()))
                X_resampled, y_resampled = SMOTE().fit_resample(x, y)
#                print(sorted(Counter(y_resampled).items()))
#                print(len(y_resampled))
                X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)
#                clf=RandomForestClassifier()
#                from sklearn import tree
#                clf = tree.DecisionTreeClassifier()
#                from sklearn.naive_bayes import GaussianNB
#                clf = GaussianNB()
#                from sklearn.svm import SVC
#                clf = SVC(gamma='auto')
                
#                from sklearn.linear_model import LogisticRegression
#                clf = LogisticRegression(C=1e5)
                
#                from sklearn.neighbors import KNeighborsClassifier
#                clf = KNeighborsClassifier(n_neighbors=5)
                from sklearn import linear_model
                clf = linear_model.BayesianRidge()
                
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                
                out_file.write("\n\n"+"model: "+model_name+"\n")
                out_file.write("Accuracy: "+ str(metrics.accuracy_score(y_test, y_pred))+"\n")
                out_file.write("Recall: "+ str(metrics.recall_score(y_test, y_pred))+"\n")
#                print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
#                print("Recall: ",metrics.recall_score(y_test, y_pred))
                conf_matrix = metrics.confusion_matrix(y_test, y_pred)
                out_file.write(str(conf_matrix))
                out_file.write("\n")
                out_file.write(str(perf_measure(y_test, y_pred)))
                out_file.write("\n")
                out_file.flush()
#                print(conf_matrix)
#                print(perf_measure(y_test, y_pred)) 
                
                import seaborn as sn
                import pandas as pd
                import matplotlib.pyplot as plt
                
                df_cm = pd.DataFrame(conf_matrix, range(2), range(2))
                plt.figure(figsize = (12,8))
                sn.set(font_scale=1.4)#for label size
                sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 28})# font size
                
                plt.savefig("/home/masudulhasanmasudb/Music/hdd_data/result_2017/"+day_number+"/"+model_name+"_bayesian.png")
                
                count+=1
#                if(count==2):
#                    break
        except:
#            print("error")
            traceback.print_exc()
#    if(count==2):
#        break
































           