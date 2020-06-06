#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 03:40:19 2020

@author: masudulhasanmasudb
"""
#import time
#import glob,random
#import datetime
#import os
#import subprocess
#import shlex
#import gc,sys, traceback
#import pandas as pd
#import collections
#
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
#parent_folder_name = "/content/drive/My Drive/dataset_2017/1/"
#folder_list=glob.glob(parent_folder_name+"*")
#for x in folder_list:
#  if "ST4000DM000.csv" in x:
#    hdd = pd.read_csv(x, header=None)
#    print(hdd.head)
#hdd = pd.read_csv(x, header=None)
##    print(hdd.head)
#
#hdd = hdd.drop(hdd.columns[6], axis=1)
#hdd = hdd.drop(hdd.columns[9], axis=1)
#hdd = hdd.drop(hdd.columns[14], axis=1)
#hdd = hdd.drop(hdd.columns[13], axis=1)
##print(hdd.describe())
#
#hdd_pos = hdd.loc[hdd[hdd.columns[12]] == 1]
##print(len(hdd_pos))
#
#hdd_neg = hdd[hdd[hdd.columns[12]] == 0]
##print(len(hdd_neg))
#
##print(len(hdd))
#del hdd
#gc.collect()
#if len(hdd_neg)!=0 and len(hdd_pos)!=0:
#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
#from sklearn import ensemble, metrics 
#import gc
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
#from sklearn import metrics
#from imblearn.over_sampling import (RandomOverSampler, SMOTE, ADASYN)
#from collections import Counter
#
#pos_train, pos_test = train_test_split(hdd_pos, test_size=0.2)
#
##print(len(pos_train))
##print(len(pos_test))
#
#neg_train, neg_test = train_test_split(hdd_neg, test_size=0.2)
##print(len(neg_train))
##print(len(neg_test))
#
#train_merged = [pos_train, neg_train]
#train_dataset = pd.concat(train_merged)
## print(Counter(X_train))
##print(train_dataset.describe)
#
#test_merged = [pos_test, neg_test]
#test_dataset = pd.concat(test_merged)
## print(Counter(X_train))
##print(test_dataset.describe)
#
#train_dataset = train_dataset.dropna()
##print(train_dataset.describe)
#
#x = train_dataset.iloc[:, :-1].values
#y = train_dataset.iloc[:, -1].values
#
#X_resampled, y_resampled = SMOTE().fit_resample(x, y)
##print(sorted(Counter(y_resampled).items()))
##print(len(y_resampled))
#
#clf=RandomForestClassifier()
#clf.fit(X_resampled,y_resampled)
#
#test_dataset = test_dataset.dropna()
##print(test_dataset.describe)
#
#X_test = test_dataset.iloc[:, :-1].values
#y_test = test_dataset.iloc[:, -1].values
#
#y_pred=clf.predict(X_test)
#
#print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
#print("Recall: ",metrics.recall_score(y_test, y_pred))
#conf_matrix = metrics.confusion_matrix(y_test, y_pred)
#
#print(conf_matrix)
#print(perf_measure(y_test, y_pred))
#
#
#del hdd_pos
#del hdd_neg
#del train_dataset
#del test_dataset
#gc.collect()


import time
import glob,random
import datetime
import os
import subprocess
import shlex
import gc,sys, traceback
import pandas as pd
import collections

#day = int(sys.argv[1])
day =1
#print(day)
def all_same(items):
    return all(x == 0 for x in items)

def get_label_list(df_list, day):
    label_list=[]
    for i in range(len(df_list)-1):
        if df_list[i+1]!= df_list[i]:
            label_list.append(1)
        else:
           label_list.append(0)
    
    final_label = []       
    for i in range(len(label_list)-day+1):
        selected_list = label_list[i:i+day]
        if all_same(selected_list):
            final_label.append(0)
        else:
            final_label.append(1)
    return final_label

parent_folder_name = "/home/masudulhasanmasudb/eclipse-workspace/Proactive_error_detection/disk_model_files/"
folder_list=glob.glob(parent_folder_name+"*")
#print(folder_list)

features = [1, 4, 5, 7, 9, 12, 187, 188, 193, 194, 196, 197, 198, 199]
columns_specified = []
for feature in features:
    	columns_specified += ["smart_{0}_raw".format(feature)]
#columns_specified = ["serial_number", "date", "model", "failure"] + columns_specified
columns_specified = ["date","model","failure"] + columns_specified
#print(columns_specified)
#sample_data = pd.DataFrame(columns=columns_specified)
#print(sample_data)

#for disk_model in disk_models:
#output_file = open("../dataset_"+disk_model+".csv","a+")
count = 0
for day in range(day,day+1):
    for x in folder_list:
        try:
#            print(x)
            df = pd.read_csv(x, header=None)
#            df = df[columns_specified]
#            print(df)
            model_name = df[2][0]
            print(model_name)
            if "ST12000NM0007" in model_name:
                bit_error_list = df[10].tolist()
                bit_error_labels= get_label_list(bit_error_list, day)
                
                last_row = len(df)-1
                
                diff = len(df_list) - len(label_list)
                
                df = df[:-diff]
                df['bit_error_label'] = bit_error_labels
                output_file = open("../"+model_name+".csv","a+")
                df.to_csv(output_file, header=False, index=False)
            
                count+=1
                print(count)
#                if(count==3):
#                    break
            del df
            gc.collect()
        except:
            print(x)
            traceback.print_exc()

