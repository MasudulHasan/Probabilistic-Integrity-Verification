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

disk_model_name = "ST4000DM000"
#disk_model_name = "ST8000DM002"
#disk_model_name = "ST8000NM0055"
#disk_model_name = "ST12000NM0007"
#disk_model_name = "ST6000DX000" //error
#disk_model_name = "ST10000NM0086" //error

number_of_days = 1

map_list = []
index_map ={}
date_dict={}
now = time.time()
count=-1
file_name=""
with open("/home/masudulhasanmasudb/eclipse-workspace/Proactive_error_detection/map_2018.txt","r")as in_file:
        for line in in_file:
            if(len(line.strip())>0):
                if".csv" in line:
                    if(count!=-1):
                        map_list.append(date_dict)
                        index_map[file_name]=count
                    
                    date_dict.clear()
                    count+=1
                    file_name=line.strip()
                else:
                    parts = line.strip().split(" ")
                    date_dict[parts[0]] = int(parts[1])
                    
#                print(line)
                
                
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
    next_days_label =[]
    
    now = datetime.datetime(year,month,day)
    next_days_list = []
    for i in range(number_of_days): 
        now += datetime.timedelta(days=1)
        next_days_list.append(now.strftime('%Y-%m-%d'))
        
    
    for x in serial_number_list:
        did_happeden = False
        index = index_map[x+".csv"]
        for day in next_days_list:
            next_day_value = map_list[index][day]
            if next_day_value ==1:
                did_happeden = True
                break
            
        if did_happeden:
            next_days_label.append(1)
        else:
            next_days_label.append(0)
        
    return next_days_label
        

def Sort_Tuple(tup):   
    return(sorted(tup, key = lambda x: x[0], reverse = True))


def calculate_accuracy(tuple_list, real_list):
    total = 0
    tf =0
    for x in range(len(tuple_list)):
        item = tuple_list[x]
        prob = float(item[0])
        if(prob<0.4):
            total+=1
            index = int(item[0])
            if real_list[index]==0:
                tf+=1
    
#    print(" calcuated accuracy = "+str((tf/total))) 

    return str((tf/total)), str(len(tuple_list)-total)           

           
hdd = pd.read_csv('../dataset_2017/dataset_'+str(disk_model_name)+'.csv')
#hdd = pd.read_csv("../dataset_1.csv")
hdd = hdd.drop(hdd.columns[6], axis=1)
hdd = hdd.drop(hdd.columns[9], axis=1)
hdd = hdd.dropna()
hdd_pos = hdd.loc[hdd[hdd.columns[12]] == 1]
hdd = hdd[hdd[hdd.columns[12]] == 0]
hdd_neg = hdd
hdd_merged = [hdd_pos, hdd_neg]
result = pd.concat(hdd_merged)
x = result.iloc[:, :-1].values
y = result.iloc[:, -1].values
X_resampled, y_resampled = SMOTE().fit_resample(x, y)
#X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)
clf=RandomForestClassifier()
clf.fit(X_resampled,y_resampled)


features = [1, 4, 5, 7, 9, 12, 188, 193, 194, 197, 198, 199]
columns_specified = []
for feature in features:
    	columns_specified += ["smart_{0}_raw".format(feature)]

stripe_size = 50

#output_file = open("final_log_19_11_19_"+str(start_year)+".txt","a+")
output_file = open("../final_logs1/"+str(number_of_days)+"/"+str(disk_model_name)+".txt","a+")
#
year = 2018
end_year = 2018
while year<=end_year:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    for month in range(1,7):
        for day in range(1,32):
            if month<=9:
                    month_str = "0"+str(month)
            else:
                month_str = str(month)
            
            if day<=9:
                day_str = "0"+str(day)
            else:
                day_str = str(day)
            
            import datetime
            correctDate = None
            try:
                newDate = datetime.datetime(year,month,day)
                correctDate = True
            except ValueError:
                correctDate = False
#            print(str(correctDate))
            
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
                    
                    print(df.shape)
                    shape = df.shape
                    print(shape[0])
                    
                    if shape[0]!=0:
                    
                        total_disk_number = 0
                        total_check_disk = 0
    
                        output_str =""
                        for numof_iter in range(400):
                            try:
                        
                                file_size =int(random.getrandbits(9))
                                output_str += "file size "+ str(file_size)+"\n"
                                disk_number = int((file_size*1024)/500)
#                                total_disk_number+=disk_number
                                if(disk_number> shape[0]):
                                    selected_disk = df
                                    total_disk_number+=shape[0]
                                else:
                                    selected_disk = df.sample(disk_number)
                                    total_disk_number+=disk_number
                                serial_number = selected_disk.iloc[:, 1].values
                                selected_disk = selected_disk[columns_specified]
                                pred_value = clf.predict(selected_disk)
                                preds = clf.predict_proba(selected_disk)
                                predicted_pair_list = []
                                
                                for x in range(len(preds[:,1])):
                                    predicted_pair_list.append((preds[:,1][x],x))
                                
                                s_list = Sort_Tuple(predicted_pair_list)
                                
                                output_str+="\n\n"+str(date)+"\n"
#                                output_str+="predicted_value: \n"
#                                output_str+=str(pred_value) + "\n" 
                                next_day_label = get_lable(serial_number,date,year,month,day)
    
#                                output_str+="next day real value: \n"
#                                output_str+=str(next_day_label)+"\n"
                                
                                c_auuracy, checksum_disk = calculate_accuracy(s_list, next_day_label)
                                total_check_disk+=int(checksum_disk)
    
                                output_str += "self calculated accuracy "+ c_auuracy+"\n"
                                output_str += "cheksem run on "+ checksum_disk +"\n"
                                TP, FP, TN, FN = perf_measure(next_day_label, pred_value)
                                output_str+="next day stat: \n"
                                output_str+= str(metrics.accuracy_score(next_day_label, pred_value))+"\n"
                                output_str+=str(metrics.recall_score(next_day_label, pred_value))+"\n"
                                output_str+="TP: "+str(TP)+" "+str(FP)+" "+str(TN)+" "+str(FN)+"\n"
    
                            except:
    #                            print("error")
                                traceback.print_exc()
    
                        output_file.write(output_str)
                        output_file.write("total_disk "+ str(total_disk_number) +"\n")
                        output_file.write("check Sum run on "+ str(total_check_disk) +"\n")
                        output_file.write("perct "+ str(total_check_disk/total_disk_number) +"\n")
                        output_file.flush()
                    del(df)
                    gc.collect()
                except:
                    traceback.print_exc()
                

    year+=1                   