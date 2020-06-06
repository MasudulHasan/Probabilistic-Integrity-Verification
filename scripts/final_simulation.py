#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 19:35:47 2019

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
import sys, traceback
import threading
import datetime

number_of_days = 0

map_list = []
index_map ={}
date_dict={}
now = time.time()
count=-1
file_name=""
with open("../map_2019.txt","r")as in_file:
    for line in in_file:
        if(len(line.strip())>0):
            if".csv" in line:
                if(count!=-1):
                    map_list.append(date_dict)
                    index_map[file_name]=count
                    if "ZA13R2LZ" in file_name:
                        print(count)
                        print(date_dict)
                
                date_dict={}
                count+=1
                file_name=line.strip()
            else:
                parts = line.strip().split(" ")
                date_dict[parts[0].strip()] = int(float(parts[1].strip()))
                    
#                print(line)

map_list.append(date_dict)
index_map[file_name]=count                   
                
print(count)


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
             
    return(TP, FP, TN, FN)

def count_unique(keys):
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return uniq_keys, np.bincount(bins)


def get_lable(serial_number_list,date,year,month,day):
    global parent_folder_name
    next_day_label ={}
    
    now = datetime.datetime(year,month,day)
    next_day = (now + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    
    for x in serial_number_list:
        index = index_map[x+'.csv']
        try:
            current_value = map_list[index][date]
            
            next_day_value = map_list[index][next_day]
            
            if(current_value==next_day_value):
                next_day_label[x]=0
            else:
    #            print("current "+str(current_value)+ " next "+str(next_day_value))
                next_day_label[x]=1
        except:
            print("value not found "+str(x)+" "+str(date))
        
    return next_day_label
        

def Sort_Tuple(tup):   
    return(sorted(tup, key = lambda x: x[0], reverse = True))


def get_prob_value(serial_list, all_prob_map, all_value_map):
    prob_list=[]
    value_list=[]
    for x in serial_list:
        try:
            prob_list.append(all_prob_map[x])
            value_list.append(all_value_map[x])
        except:
            print(x)
    
    return prob_list, value_list


def get_threshold(predicted_value, prob_list, real_values):
    threshold_list = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]
    
    final_string=""
    
    for base_threshold in threshold_list:
        tp=0 
        tn=0
        fp=0
        fn=0
        
        
        for x in range(len(prob_list)):
            prob = float(prob_list[:,1][x])
            if prob >= base_threshold:
                if real_values[x]==1:
                  tp+=1
                else:
                  fp+=1
            else:
                if predicted_value[x]== 1 and real_values[x]==1:
                  tp+=1
                elif predicted_value[x]== 0 and real_values[x]==1:
                  fn+=1
                
                elif predicted_value[x]== 0 and real_values[x]==0:
                  tn+=1
                elif predicted_value[x]== 1 and real_values[x]==0:
                  fp+=1
        
        recall = int(tp/(tp+fn))
        if recall==1:
             return base_threshold


def calculate_accuracy(predicted_value, prob_list, real_values, threshold):
    threshold_list = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]
    
    calculated_threshold=0.0
    
    for base_threshold in threshold_list:
        tp=0 
        tn=0
        fp=0
        fn=0
        
        for x in range(len(prob_list)):
            prob = float(prob_list[:,1][x])
            if prob >= base_threshold:
                if real_values[x]==1:
                  tp+=1
                else:
                  fp+=1
            else:
                if predicted_value[x]== 1 and real_values[x]==1:
                  tp+=1
                elif predicted_value[x]== 0 and real_values[x]==1:
                  fn+=1
                
                elif predicted_value[x]== 0 and real_values[x]==0:
                  tn+=1
                elif predicted_value[x]== 1 and real_values[x]==0:
                  fp+=1
                  

        recall = int(tp/(tp+fn))
        if recall==1:
            calculated_threshold = base_threshold
            break
    
    if calculated_threshold == threshold:
        return True
    return False
             
selected_models = ['ST8000NM0055']

out_file = open("../final_simulation_result.txt", "a+")

for disk_model_name in selected_models:
    
    for number_of_days in range(0,6):
        hdd_extra = pd.read_csv("../final_dataset/"+str(1)+'/'+str(disk_model_name)+'.csv', header=None)
        print(hdd_extra.head())
        hdd_extra = hdd_extra.drop(hdd_extra.columns[0], axis=1)
        hdd_extra = hdd_extra.drop(hdd_extra.columns[6], axis=1)
        hdd_extra = hdd_extra.drop(hdd_extra.columns[9], axis=1)
        hdd_extra = hdd_extra.drop(hdd_extra.columns[14], axis=1)
        hdd_extra = hdd_extra.drop(hdd_extra.columns[13], axis=1)
        hdd_extra = hdd_extra.dropna()
        result = hdd_extra
        print(result.head())
        
        x = result.iloc[:, :-1].values
        y = result.iloc[:, -1].values
        
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler()
        del hdd_extra
        del result
        gc.collect()
        
        X_resampled, y_resampled = rus.fit_resample(x, y)
        print(Counter(y_resampled))
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)
        clf=RandomForestClassifier()
        clf.fit(X_train,y_train)
        all_pred_value = clf.predict(X_test)
        prob_list = clf.predict_proba(X_test)
        
        base_threshold = get_threshold(all_pred_value, prob_list, y_test)
        
        
        hdd = pd.read_csv("../one_month_files/"+str(number_of_days)+'/'+str(disk_model_name)+'.csv', header=None)
        hdd = hdd.drop(hdd.columns[0], axis=1)
        hdd = hdd.drop(hdd.columns[6], axis=1)
        hdd = hdd.drop(hdd.columns[9], axis=1)
        hdd = hdd.drop(hdd.columns[14], axis=1)
        hdd = hdd.drop(hdd.columns[13], axis=1)
        hdd = hdd.dropna()
        result = hdd
        print(result.head())
        
        x = result.iloc[:, :-1].values
        y = result.iloc[:, -1].values
        
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler()
        del hdd
        del result
        gc.collect()
        
        X_resampled, y_resampled = rus.fit_resample(x, y)
        print(Counter(y_resampled))
        clf=RandomForestClassifier()
        clf.fit(X_resampled,y_resampled)
        
        
        hdd_test = pd.read_csv("../one_month_files/"+str(number_of_days+1)+'/'+str(disk_model_name)+'.csv', header=None)
        hdd_test = hdd_test.drop(hdd_test.columns[0], axis=1)
        hdd_test = hdd_test.drop(hdd_test.columns[6], axis=1)
        hdd_test = hdd_test.drop(hdd_test.columns[9], axis=1)
        hdd_test = hdd_test.drop(hdd_test.columns[14], axis=1)
        hdd_test = hdd_test.drop(hdd_test.columns[13], axis=1)
        hdd_test = hdd_test.dropna()
        result = hdd_test
        print(result.head())
        
        x = result.iloc[:, :-1].values
        y = result.iloc[:, -1].values
        
        del hdd_test
        del result
        gc.collect()
        
        all_pred_value = clf.predict(x)
        prob_list = clf.predict_proba(x)
        
        log_file = open("../final_simulation_logs/"+str(number_of_days+1)+".txt", "a+")
        
        for i in range(len(all_pred_value)):
           log_file.write(str(all_pred_value[i])+" "+str(y[i])+" "+str(prob_list[:,1][i])+"\n") 
        
        
        
        if calculate_accuracy(all_pred_value, prob_list, y, base_threshold):
            out_file.write(str(number_of_days+1)+" -> yes\n")
        else:
            out_file.write(str(number_of_days+1)+" -> no\n")
        

    
    
    
    