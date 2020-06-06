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
with open("../map_2019.txt","r")as in_file:
    for line in in_file:
        if(len(line.strip())>0):
            if".csv" in line:
#                print(line)
                if(count!=-1):
                    map_list.append(date_dict)
                    index_map[file_name]=count
#                    if "ZA13R2LZ" in file_name:
#                        print(count)
#                        print(date_dict)
                
                date_dict={}
                count+=1
                file_name=line.strip()
            else:
                parts = line.strip().split(" ")
                date_dict[parts[0].strip()] = int(float(parts[1].strip()))
                    
#                print(line)

#print(file_name)
#print(date_dict)
map_list.append(date_dict)
index_map[file_name]=count                
                
print(count)
#print(len(map_list))


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
#    return(TP, FP, TN, FN)
#
#def count_unique(keys):
#    uniq_keys = np.unique(keys)
#    bins = uniq_keys.searchsorted(keys)
#    return uniq_keys, np.bincount(bins)
#
#
#def get_lable(serial_number_list,date,year,month,day):
#    global parent_folder_name
#    next_day_label ={}
#    
#    now = datetime.datetime(year,month,day)
#    next_day = (now + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
#    
#    for x in serial_number_list:
#        index = index_map[x+'.csv']
#        try:
#            current_value = map_list[index][date]
#            
#            next_day_value = map_list[index][next_day]
#            
#            if(current_value==next_day_value):
#                next_day_label[x]=0
#            else:
#    #            print("current "+str(current_value)+ " next "+str(next_day_value))
#                next_day_label[x]=1
#        except:
#            print("value not found "+str(x)+" "+str(date))
#        
#    return next_day_label
#        
#
#def Sort_Tuple(tup):   
#    return(sorted(tup, key = lambda x: x[0], reverse = True))
#
#
#def get_prob_value(serial_list, all_prob_map, all_value_map):
#    prob_list=[]
#    value_list=[]
#    for x in serial_list:
#        try:
#            prob_list.append(all_prob_map[x])
#            value_list.append(all_value_map[x])
#        except:
#            print(x)
#    
#    return prob_list, value_list
#
#
#def calculate_accuracy(prob_map, value_map, real_value_map):
#    threshold_list = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]
#    
#    final_string=""
#    
#    for base_threshold in threshold_list:
#        tp=0 
#        tn=0
#        fp=0
#        fn=0
#        
#        
#        for x in real_value_map.keys():
#            prob = float(prob_map[x])
#            if prob >= base_threshold:
#                if real_value_map[x]==1:
#                  tp+=1
#                else:
#                  fp+=1
#            else:
#                if value_map[x]== 1 and real_value_map[x]==1:
#                  tp+=1
#                elif value_map[x]== 0 and real_value_map[x]==1:
#                  fn+=1
#                
#                elif value_map[x]== 0 and real_value_map[x]==0:
#                  tn+=1
#                elif value_map[x]== 1 and real_value_map[x]==0:
#                  fp+=1
#                  
##        print("TP, FP, TN, FN = "+str(tp)+" "+str(fp)+" "+str(tn)+" "+str(fn))
#
#        final_string+="\n\nThreshold "+str(base_threshold)+"\n"
#        final_string+="TP, FP, TN, FN = "+str(tp)+" "+str(fp)+" "+str(tn)+" "+str(fn)+"\n"
#        try:
#            final_string+="Recall: "+ str(tp/(tp+fn))+"\n"
#        except:
#            final_string+="Recall: "+ str(0)+"\n"
#        try:
#            final_string+="extra: "+ str((fp/(tn+fp))*100)+"\n"
#        except:
#            final_string+="extra: "+ str(0)+"\n"
#    return final_string
#             
#selected_models = ['ST12000NM0007']
##selected_models = ['ST12000NM0007']
#
#for disk_model_name in selected_models:
#
#    hdd = pd.read_csv("../final_dataset/"+str(number_of_days)+'/'+str(disk_model_name)+'.csv', header=None)
#    #hdd = pd.read_csv("../dataset_1.csv")
#    hdd = hdd.drop(hdd.columns[6], axis=1)
#    hdd = hdd.drop(hdd.columns[9], axis=1)
#    hdd = hdd.drop(hdd.columns[14], axis=1)
#    hdd = hdd.drop(hdd.columns[13], axis=1)
#    hdd = hdd.dropna()
##    print(hdd.head())
#    
#    hdd_extra = pd.read_csv("../2019_files/"+str(number_of_days)+'/'+str(disk_model_name)+'.csv', header=None)
##    print(hdd_extra.head())
#    hdd_extra = hdd_extra.drop(hdd_extra.columns[6], axis=1)
#    hdd_extra = hdd_extra.drop(hdd_extra.columns[9], axis=1)
#    hdd_extra = hdd_extra.drop(hdd_extra.columns[14], axis=1)
#    hdd_extra = hdd_extra.drop(hdd_extra.columns[13], axis=1)
#    hdd_extra = hdd_extra.dropna()
#    
#    hdd_merged = [hdd, hdd_extra]
#    result = pd.concat(hdd_merged)
##    print(result.head())
#    #result = result.dropna()
#    
#    x = result.iloc[:, :-1].values
#    y = result.iloc[:, -1].values
#    
#    from imblearn.under_sampling import RandomUnderSampler
#    rus = RandomUnderSampler()
#    
#    del hdd
#    del hdd_extra
#    del hdd_merged
#    del result
#    gc.collect()
#    
#    X_resampled, y_resampled = rus.fit_resample(x, y)
#    print(Counter(y_resampled))
##    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)
#    clf=RandomForestClassifier()
#    clf.fit(X_resampled,y_resampled)
#    
#    
#    features = [1, 4, 5, 7, 9, 12, 188, 193, 194, 197, 198, 199]
#    columns_specified = []
#    for feature in features:
#        	columns_specified += ["smart_{0}_raw".format(feature)]
#    
#    main_columns_specified = ["serial_number"] + columns_specified
#    
#    error_disks = ['ZCH0CM1R', 'ZCH06HRT', 'ZJV03KCP', 'ZCH09PKH', 'ZJV005J2', 'ZCH0D5CK', 'ZCH0CY4F', 'ZCH09LGZ', 'ZCH073S9', 'ZJV03NCV', 'ZJV1CGQL', 'ZCH066F9', 'ZJV2ECQW', 'ZCH0D2GD', 'ZCH06WWK', 'ZCH06N6B', 'ZCH0ADNJ', 'ZJV00A05', 'ZCH0D2BE', 'ZCH0CFN9', 'ZCH0C0ZV', 'ZJV03N4D', 'ZCH0BHPD', 'ZCH0CHNQ', 'ZCH070B8', 'ZCH06GPN', 'ZJV02C2X', 'ZJV10J7D', 'ZCH06V2F', 'ZCH07CAW', 'ZJV00DK9', 'ZCH0AQ1Q', 'ZCH07WBW', 'ZCH08XHJ', 'ZJV2FQWM', 'ZCH06MSQ', 'ZJV1C4CN', 'ZCH07H06', 'ZCH081NG', 'ZCH089LH', 'ZCH0CZ28', 'ZCH0D6ZB', 'ZCH0CZNE', 'ZJV12RY2', 'ZJV2E8LZ', 'ZCH080V2', 'ZCH06FXX', 'ZCH07RK0', 'ZCH0DG3H', 'ZCH03Y6Y', 'ZCH07F30', 'ZCH0BTAS', 'ZJV2VFCC', 'ZCH05F0M', 'ZCH0BWPG', 'ZJV00FRS', 'ZCH086XE', 'ZJV03KM5', 'ZCH078KR', 'ZCH0DG4X', 'ZJV2FPYG', 'ZJV19EWJ', 'ZCH06EHM', 'ZCH07ZDP', 'ZCH07XHS', 'ZJV02CR4', 'ZJV2EB6S', 'ZCH0926P', 'ZCH060TB', 'ZCH07665']
#    
#    stripe_size = 50
#    
#    output_file = open("../simulation_result/"+str(disk_model_name)+".txt","a+")
#    ##
#    year = 2019
#    end_year = 2019
#    while year<=end_year:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
#        for month in range(7,8):
#            for day in range(1,16):
#                if month<=9:
#                        month_str = "0"+str(month)
#                else:
#                    month_str = str(month)
#                
#                if day<=9:
#                    day_str = "0"+str(day)
#                else:
#                    day_str = str(day)
#                
#                correctDate = None
#                try:
#                    newDate = datetime.datetime(year,month,day)
#                    correctDate = True
#                except ValueError:
#                    correctDate = False
#                if correctDate==True:
#                    try:
#                        date = str(year)+"-"+month_str+"-"+day_str
#                        print(date)
#                        df = pd.read_csv("../data/"+date+".csv")
#    #                    df = df.loc[df['model'] == 'ST4000DM000']
#                        df = df.loc[df['model'] == str(disk_model_name)]
#                        df = df.loc[df['serial_number'] !="S300XQ5W"]
#                        df = df.loc[df['serial_number'] !="W0Q7D8BD"]
#                        df = df.loc[df['serial_number'] !="S300XCF0"]
#                        df = df[main_columns_specified]
#                        df = df.dropna()
#                        shape = df.shape
#                        if shape[0]!=0:
#                            
#                            selected_serial_number = df.iloc[:, 0].values
#                            df_selected = df[columns_specified]
#                            df_selected = df_selected.dropna()
#                            print(df_selected.shape)
#                            all_pred_value = clf.predict(df_selected)
#                            all_preds = clf.predict_proba(df_selected)
#                            
#                            prob_value_map={}
#                            predicted_value_map={}
#                            for x in range(len(all_preds[:,1])):
#                                prob_value_map[selected_serial_number[x]] = all_preds[:,1][x]
#                                predicted_value_map[selected_serial_number[x]] = all_pred_value[x]
#                                
##                            print(prob_value_map)
##                            print(predicted_value_map)
#                            
#                            total_disk_number = 0
#                            total_check_disk = 0
#                            wl = np.random.poisson(lam=1.123983e+05)
#                            print(wl)
#                            output_str =""
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
##                                    selected_disk = df
#                                    if(disk_number> shape[0]):
#                                        selected_disk = df
#                                        total_disk_number+=shape[0]
#                                    else:
#                                        selected_disk = df.sample(disk_number)
#                                        total_disk_number+=disk_number
#                                        
#                                    serial_number = selected_disk.iloc[:, 0].values
#                                    
##                                    serial_number = error_disks
#                                    
##                                    selected_disk = selected_disk[columns_specified]
##                                    pred_value = clf.predict(selected_disk)
##                                    preds = clf.predict_proba(selected_disk)
#                                    
##                                    preds, pred_value = get_prob_value(serial_number, prob_value_map, predicted_value_map)
##                                    print(preds)
##                                    print(pred_value)
#                                    
##                                    predicted_pair_list = []
#                                    
##                                    for x in range(len(preds)):
##                                        predicted_pair_list.append((preds[x],pred_value[x]))
#                                    
#                                    output_str+=str(date)+"\n"
#                                    next_day_label = get_lable(serial_number,date,year,month,day)
##                                    print(next_day_label)
#                                    output_str+=calculate_accuracy(prob_value_map, predicted_value_map, next_day_label)
#                                except:
#        #                            print("error")
#                                    traceback.print_exc()
#        
#                            output_file.write(output_str)
#                            output_file.flush()
#                        del(df)
#                        gc.collect()
#                    except:
#                        traceback.print_exc()
#                    
#    
#        year+=1                   
