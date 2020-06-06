#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:08:47 2020

@author: masudulhasanmasudb
"""
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
    
    # print(TP/(TP+FN)) 
    
    # print(FP/(FP+TN))       
           
    return(TP, FP, TN, FN)


month_to_consider = 2

selected_models = ['ST4000DM000', 'ST8000DM002', 'ST12000NM0007', 'ST8000NM0055', 'ST3000DM001', 'ST4000DX000']
out_file = open("../month_wise_result.txt","a+")
for disk_model_name in selected_models:
    year = 2016
    while(year<=2019):
        if(year==2016):
            month=month_to_consider+1
        else:
            month=1
        
        while(month<=12):
#            if(month==1):
#                train_file_path = "../month_wise_file/"+str(year-1)+"/12/"+disk_model_name+".csv"
#            else:
#                train_file_path = "../month_wise_file/"+str(year)+"/"+ str(month-1)+"/"+disk_model_name+".csv"
                
            test_file_path = "../month_wise_file/"+str(year)+"/"+ str(month)+"/"+disk_model_name+".csv"
            try:
                df_list=[]
                i =0
                year_considering = year
                month_considering = month
                while(i<month_to_consider):
                    month_considering-=1
                    if(month_considering==0):
                        month_considering = 12
                        year_considering-=1
                    
                    train_file_path = "../month_wise_file/"+str(year_considering)+"/"+ str(month_considering)+"/"+disk_model_name+".csv"
                    hdd = pd.read_csv(train_file_path, header=None)
                    hdd = hdd.drop(hdd.columns[6], axis=1)
                    hdd = hdd.drop(hdd.columns[9], axis=1)
                    df
                    
                    i+=1
                
                
                
                hdd = pd.read_csv(train_file_path, header=None)
                hdd = hdd.drop(hdd.columns[6], axis=1)
                hdd = hdd.drop(hdd.columns[9], axis=1)
                
                x_new = hdd.iloc[:, :-1].values
                y_new = hdd.iloc[:, -1].values
                from imblearn.under_sampling import RandomUnderSampler
                rus = RandomUnderSampler()
    
                X_resampled, y_resampled = rus.fit_resample(x_new, y_new)
                print(Counter(y_resampled))
      
                clf= RandomForestClassifier()
                clf.fit(X_resampled,y_resampled)
      
                test_hdd = pd.read_csv(test_file_path, header=None)
                test_hdd = test_hdd.drop(test_hdd.columns[6], axis=1)
                test_hdd = test_hdd.drop(test_hdd.columns[9], axis=1)
                test_hdd = test_hdd.dropna()
      
                X_test = test_hdd.iloc[:, :-1].values
                y_test = test_hdd.iloc[:, -1].values
                y_pred=clf.predict(X_test)
                
                print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
                print("Recall: ",metrics.recall_score(y_test, y_pred))
      
                out_file.write("\n\n"+"model: "+disk_model_name+" "+str(year)+" "+str(month)+"\n")
                out_file.write("\n")
                out_file.write("Accuracy: "+ str(metrics.accuracy_score(y_test, y_pred))+"\n")
                out_file.write("Recall: "+ str(metrics.recall_score(y_test, y_pred))+"\n")
                conf_matrix = metrics.confusion_matrix(y_test, y_pred)
                out_file.write(str(conf_matrix))
                out_file.write("\n")
                out_file.write(str(perf_measure(y_test, y_pred)))
                out_file.write("\n")
                out_file.flush()
            except:
#                print(x)
                traceback.print_exc()
                
            month+=1
        year+=1
                
