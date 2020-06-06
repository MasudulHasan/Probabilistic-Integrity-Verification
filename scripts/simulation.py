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
    
#    print(TP/(TP+FN)) 
    
#    print(FP/(FP+TN))       
           
    return(TP, FP, TN, FN)

def count_unique(keys):
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return uniq_keys, np.bincount(bins)

parent_folder_name = "/home/masudulhasanmasudb/eclipse-workspace/Proactive_error_detection/output_files/"
folder_list=glob.glob(parent_folder_name+"*")
##out_file = open("done_so_far.txt","a+")
#print(len(folder_list))


def get_lable(serial_number_list,date,year,month,day):
    global parent_folder_name
    
    next_day_label =[]
    seven_days_label = []
    
    
    now = datetime.datetime(year,month,day)
    next_days_list = []
    for i in range(1): 
        now += datetime.timedelta(days=1)
#        print(now.strftime('%Y-%m-%d') )
        next_days_list.append(now.strftime('%Y-%m-%d'))
    
    for x in serial_number_list:
        count = 0
        df = pd.read_csv(parent_folder_name+x+".csv")
        current_value = df.loc[df['date'] == date, 'smart_187_raw'].iloc[0]
#        print(current_value)
#        current_value = df['smart_187_raw']
        
        next_day_value = df.loc[df['date'] == next_days_list[0], 'smart_187_raw'].iloc[0]
        
#        print("next "+str(next_day_value))
        
        if(current_value==next_day_value):
            next_day_label.append(0)
        else:
            next_day_label.append(1)
        
#        print(next_day_label)
        
#        is_same = True
#        
#        for day in next_days_list:
#            try:
#                next_day_value = df.loc[df['date'] == day, 'smart_187_raw'].iloc[0]
#    #            print("next "+str(next_day_value))
#                if(current_value!=next_day_value):
#                    is_same = False
#                    break
#            except:
#                print("something happend "+day+" "+x)
#        
#        if is_same:
#            seven_days_label.append(0)
#        else:
#            seven_days_label.append(1)
            
        
    return next_day_label, seven_days_label
        
#        print(seven_days_label)    

#        print(len(df))
#        with open(parent_folder_name+x+".csv","r") as in_file:
#            for line in in_file:
#                print(line)
#                count+=1
#                if(count==1):
#                    break
            



hdd = pd.read_csv('dataset_1.csv')
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



error_file_list=[]
with open("files_with_error.txt","r")as in_file:
        for line in in_file:
            if(len(line.strip())>0):
                error_file_list.append(line.strip())

good_file_list = []
for x in folder_list:
    back_slash_index = x.rfind("/")
    file_name = x[back_slash_index+1:]
#    print(file_name)
    if file_name not in error_file_list:
        good_file_list.append(x)

#print(len(folder_list) - len(good_file_list)) 

test_error_file_list=[]
with open("files_for_test.txt","r")as in_file:
        for line in in_file:
            if(len(line.strip())>0):
                test_error_file_list.append(line.strip())

#print(len(test_error_file_list))

all_file_list = good_file_list+test_error_file_list
#print(len(all_file_list))
stripe_size = 50

output_file = open("final_log.txt","a+")

year = 2015; 
while year<=2017:
    for month in range(1,13):
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
                    file_size =int(random.getrandbits(7))
    #                print(file_size)
                    
                    df = pd.read_csv("data/"+date+".csv")
                    df = df.loc[df['model'] == 'ST4000DM000']
    #                print(df['model'])
    #                print(len(df))
                    
                    
                    disk_number = int((file_size*1024)/500)
    #                print(disk_number)
                    
                    selected_disk = df.sample(500)
                    serial_number = selected_disk.iloc[:, 1].values
    #                print(serial_number)
                    selected_disk = selected_disk[columns_specified]
                    pred_value = clf.predict(selected_disk)
                    preds = clf.predict_proba(selected_disk)
                    
    #                print(pred_value)
    #                print(preds)
                    
                    output_file.write("\n\n"+str(date)+"\n")
                    
                    output_file.write("predicted_value: \n")
                    output_file.write(str(pred_value)+"\n")
                    output_file.write("probability: \n")
                    output_file.write(str(preds)+"\n")
                    
                    
    #                print(selected_disk.head)
                    
                    next_day_label, seven_days_label = get_lable(serial_number,date,year,month,day)
                    output_file.write("next day real value: \n")
                    output_file.write(str(next_day_label)+"\n")
#                    output_file.write("seven days value: \n")
#                    output_file.write(str(seven_days_label)+"\n")
                    
                    output_file.write("next day stat: \n")
                    print("N Accuracy: ",metrics.accuracy_score(next_day_label, pred_value))
                    print("N Recall: ",metrics.recall_score(next_day_label, pred_value))
                    output_file.write(str(metrics.accuracy_score(next_day_label, pred_value))+"\n")
                    output_file.write(str(metrics.recall_score(next_day_label, pred_value))+"\n")
    #                conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    #                print(conf_matrix)
                    TP, FP, TN, FN = perf_measure(next_day_label, pred_value)
                    output_file.write(str(TP)+" "+str(FP)+" "+str(TN)+" "+str(FN)+"\n")
                    
                    
#                    output_file.write("seven days stat: \n")
#                    print("S Accuracy: ",metrics.accuracy_score(seven_days_label, pred_value))
#                    print("S Recall: ",metrics.recall_score(seven_days_label, pred_value))
#                    output_file.write(str(metrics.accuracy_score(seven_days_label, pred_value))+"\n")
#                    output_file.write(str(metrics.recall_score(seven_days_label, pred_value))+"\n")
#    #                conf_matrix = metrics.confusion_matrix(y_test, y_pred)
#    #                print(conf_matrix)
#                    
#                    TP, FP, TN, FN = perf_measure(seven_days_label, pred_value)
#                    output_file.write(str(TP)+" "+str(FP)+" "+str(TN)+" "+str(FN)+"\n")
#                    output_file.flush()
                
                except:
                    traceback.print_exc()
#                    print("error")
                
                
#                print(next_day_label)
#                print(seven_days_label)
#                predicted_values=[]
#                predicted_prob = []
#                for x in range (len(selected_disk)):
#                    print(selected_disk.iloc[x].values)
#                    value_list = selected_disk.iloc[x].values
#                    pred_value = clf.predict([value_list])
#                    preds = clf.predict_proba([value_list])
#                    print(pred_value[0])
##                    print(preds[:,1][0])
##                    print(preds[:,0][0])
#                    predicted_prob.append(preds[:,1][0])
#                
#                selected_disk_list = random.sample(range(0, len(all_file_list)-1), 10000)
#                
##                print(random.sample(range(0, len(all_file_list)-1), disk_number))
#                
#                selected_disk_number =0
#                disk_stat = []
#                for x in selected_disk_list:
#                    if selected_disk_number>=disk_number:
#                        break
#                    
#                    file_name = all_file_list[x]
#                    print(file_name)
#                    with open(file_name,"r")as in_file:
#                        for line in in_file:
#                            if date in line:
#                                selected_disk_number+=1
#                                print(line)
#                                disk_stat.append(line)
#                print(disk_stat)                
                
                
                
                
    year+=1                   