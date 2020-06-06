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

#disk_model_name = "ST4000DM000"
#disk_model_name = "ST8000DM002"
#disk_model_name = "ST8000NM0055"
disk_model_name = "ST12000NM0007"
#disk_model_name = "ST6000DX000" //error
#disk_model_name = "ST10000NM0086" //error

number_of_days = 1

map_list = []
index_map ={}
date_dict={}
now = time.time()
count=-1
file_name=""
#with open("../map_2019.txt","r")as in_file:
#    for line in in_file:
#        if(len(line.strip())>0):
#            if".csv" in line:
#                if(count!=-1):
#                    map_list.append(date_dict)
#                    index_map[file_name]=count
#                
#                date_dict.clear()
#                count+=1
#                file_name=line.strip()
#            else:
#                parts = line.strip().split(" ")
#                date_dict[parts[0]] = int(float(parts[1]))
#                    
##                print(line)
#                
#                
#print(count)

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
    next_day_label =[]
    
    now = datetime.datetime(year,month,day)
    next_day = (now + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    
    for x in serial_number_list:
        index = index_map[x+".csv"]
        current_value = map_list[index][date]
        
        next_day_value = map_list[index][next_day]
        
        if(current_value==next_day_value):
            next_day_label.append(0)
        else:
            next_day_label.append(1)
        
    return next_day_label
        

def Sort_Tuple(tup):   
    return(sorted(tup, key = lambda x: x[0], reverse = True))


def calculate_accuracy(tuple_list, real_list):
    threshold_list = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]
    
    final_string=""
    
    for base_threshold in threshold_list:
        tp=0 
        tn=0
        fp=0
        fn=0
        for x in range(len(tuple_list)):
            item = tuple_list[x]
            prob = float(item[0])
#            print(prob)
#            print(real_list[x])
#            print(item[1])
#            print(base_threshold)
            if prob >= base_threshold:
                if real_list[x]==1:
                  tp+=1
                else:
                  fp+=1
            else:
                if item[1]== 1 and real_list[x]==1:
                  tp+=1
                elif item[1]== 0 and real_list[x]==1:
                  fn+=1
                
                elif item[1]== 0 and real_list[x]==0:
                  tn+=1
                elif item[1]== 1 and real_list[x]==0:
                  fp+=1
                  
#        print("TP, FP, TN, FN = "+str(tp)+" "+str(fp)+" "+str(tn)+" "+str(fn))

        final_string+="\n\nThreshold "+str(base_threshold)+"\n"
        final_string+="TP, FP, TN, FN = "+str(tp)+" "+str(fp)+" "+str(tn)+" "+str(fn)+"\n"
        try:
            final_string+="Recall: "+ str(tp/(tp+fn))+"\n"
        except:
            final_string+="Recall: "+ str(0)+"\n"
        try:
            final_string+="extra: "+ str((fp/(tn+fp))*100)+"\n"
        except:
            final_string+="extra: "+ str(0)+"\n"
    return final_string
             
#selected_models = ['ST4000DM000', 'ST8000DM002', 'ST12000NM0007', 'ST8000NM0055', 'ST3000DM001', 'ST4000DX000']
selected_models = ['ST4000DX000']

for disk_model_name in selected_models:

    hdd = pd.read_csv("../final_dataset/"+str(number_of_days)+'/'+str(disk_model_name)+'.csv', header=None)
    #hdd = pd.read_csv("../dataset_1.csv")
    hdd = hdd.drop(hdd.columns[6], axis=1)
    hdd = hdd.drop(hdd.columns[9], axis=1)
    hdd = hdd.drop(hdd.columns[14], axis=1)
    hdd = hdd.drop(hdd.columns[13], axis=1)
    hdd = hdd.dropna()
#    print(hdd.head())
    
    hdd_extra = pd.read_csv("../2019_files/"+str(number_of_days)+'/'+str(disk_model_name)+'.csv', header=None)
#    print(hdd_extra.head())
    hdd_extra = hdd_extra.drop(hdd_extra.columns[6], axis=1)
    hdd_extra = hdd_extra.drop(hdd_extra.columns[9], axis=1)
    hdd_extra = hdd_extra.drop(hdd_extra.columns[14], axis=1)
    hdd_extra = hdd_extra.drop(hdd_extra.columns[13], axis=1)
    hdd_extra = hdd_extra.dropna()
    
    hdd_merged = [hdd, hdd_extra]
    result = pd.concat(hdd_merged)
#    print(result.head())
    #result = result.dropna()
    
    x = result.iloc[:, :-1].values
    y = result.iloc[:, -1].values
    
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler()
    
    del hdd
    del hdd_extra
    del hdd_merged
    del result
    gc.collect()
    
    X_resampled, y_resampled = rus.fit_resample(x, y)
    print(Counter(y_resampled))
#    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)
    clf=RandomForestClassifier()
    clf.fit(X_resampled,y_resampled)
    
    
    features = [1, 4, 5, 7, 9, 12, 188, 193, 194, 197, 198, 199]
    columns_specified = []
    for feature in features:
        	columns_specified += ["smart_{0}_raw".format(feature)]
    
    stripe_size = 50
    
    output_file = open("../result_log/"+str(disk_model_name)+".txt","a+")
    ##
    year = 2019
    end_year = 2019
    while year<=end_year:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        for month in range(7,8):
            for day in range(1,16):
                if month<=9:
                        month_str = "0"+str(month)
                else:
                    month_str = str(month)
                
                if day<=9:
                    day_str = "0"+str(day)
                else:
                    day_str = str(day)
                
                correctDate = None
                try:
                    newDate = datetime.datetime(year,month,day)
                    correctDate = True
                except ValueError:
                    correctDate = False
                if correctDate==True:
                    try:
                        date = str(year)+"-"+month_str+"-"+day_str
                        print(date)
                        df = pd.read_csv("../data/"+date+".csv")
    #                    df = df.loc[df['model'] == 'ST4000DM000']
                        df = df.loc[df['model'] == str(disk_model_name)]
                        df = df.loc[df['serial_number'] !="S300XQ5W"]
                        df = df.loc[df['serial_number'] !="W0Q7D8BD"]
                        df = df.loc[df['serial_number'] !="W3004WHH"]
                        shape = df.shape
                        if shape[0]!=0:
                            
                            selected_serial_number = df.iloc[:, 1].values
                            df_selected = df[columns_specified]
                            all_pred_value = clf.predict(df_selected)
                            all_preds = clf.predict_proba(df_selected)
                            
                            prob_value_map={}
                            predicted_value_map={}
                            for x in range(len(all_preds[:,1])):
                                prob_value_map[selected_serial_number[x]] = all_preds[:,1][x]
                                predicted_value_map[selected_serial_number[x]] = all_pred_value[x]
                                
                            
                            
                            total_disk_number = 0
                            total_check_disk = 0
                            wl = np.random.poisson(lam=1.123983e+05)
                            print(wl)
                            output_str =""
#                            for numof_iter in range(wl):
#                                try:
#                                    s = np.random.uniform(0,1)
#                                    if s>.9:
#                                        file_size = np.random.poisson(lam=1.165580e+07)
#                                    else:
#                                        file_size = np.random.poisson(lam=2.082032e+01)
#                                    
#                                    output_str += "\n\nfile size "+ str(file_size)+"\n"
#                                    disk_number = int((file_size/1024)/50)+1
#                                    
#                                    if(disk_number> shape[0]):
#                                        selected_disk = df
#                                        total_disk_number+=shape[0]
#                                    else:
#                                        selected_disk = df.sample(disk_number)
#                                        total_disk_number+=disk_number
#                                        
#                                    serial_number = selected_disk.iloc[:, 1].values
#                                    selected_disk = selected_disk[columns_specified]
#                                    pred_value = clf.predict(selected_disk)
#                                    preds = clf.predict_proba(selected_disk)
#                                    predicted_pair_list = []
#                                    
#                                    for x in range(len(preds[:,1])):
#                                        predicted_pair_list.append((preds[:,1][x],pred_value[x]))
#                                    
#                                    output_str+=str(date)+"\n"
#                                    next_day_label = get_lable(serial_number,date,year,month,day)
#                                    output_str+=calculate_accuracy(predicted_pair_list, next_day_label)
#                                except:
#        #                            print("error")
#                                    traceback.print_exc()
#        
#                            output_file.write(output_str)
#                            output_file.flush()
                        del(df)
                        gc.collect()
                    except:
                        traceback.print_exc()
                    
    
        year+=1                   
