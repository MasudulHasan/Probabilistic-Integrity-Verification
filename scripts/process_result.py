#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:40:56 2020

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

parent_folder_name = "../simulation_result/with_one_month/"
folder_list=glob.glob(parent_folder_name+"*")
#print(folder_list)

#all_value_list=[]
count=0
for x in folder_list:
    if ".png" not in x:
        all_value_list=[]
        start_index = x.rfind("/")
        end_index = x.rfind(".txt")
        model_name = x[start_index+1:end_index]
        print(model_name)
        with open(x,"r+") as in_file:
            for line in in_file:
                if len(line.strip())>0:
#                    count+=1
                    if "Threshold" in line:
                        all_value_list.append(line)
                    if "TP, FP, TN, FN" in line:
                        all_value_list.append(line)
        
        fp_map={}
        tp_map={}
        fp_count_map={}
        tp_count_map={}

        for x in range(len(all_value_list)):
            if "Threshold" in all_value_list[x]:
                parts = all_value_list[x].split(" ")
                threshold_value = float(parts[1].strip())

                value_parts = all_value_list[x+1].split(" ")
                tp = int(value_parts[5].strip())
                fp = int(value_parts[6].strip())
                tn = int(value_parts[7].strip())
                fn = int(value_parts[8].strip())

                if(tp+fn!=0):
                    recall = tp/(tp+fn)

                    if threshold_value not in tp_map:
                        tp_map[threshold_value] = recall
                    else:
                        prev_value = tp_map[threshold_value]
                        tp_map[threshold_value] = prev_value + recall

                    if threshold_value not in tp_count_map:
                        tp_count_map[threshold_value] = 1
                    else:
                        prev_value = tp_count_map[threshold_value]
                        tp_count_map[threshold_value] = prev_value + 1



                if(tn+fp!=0):
                    extra = fp/(tn+fp)

                    if threshold_value not in fp_map:
                        fp_map[threshold_value] = extra
                    else:
                        prev_value = fp_map[threshold_value]
                        fp_map[threshold_value] = prev_value + extra

                    if threshold_value not in fp_count_map:
                        fp_count_map[threshold_value] = 1
                    else:
                        prev_value = fp_count_map[threshold_value]
                        fp_count_map[threshold_value] = prev_value + 1
                        
        x_values = []
        fp_values = []
        for x in fp_count_map.keys():
            x_values.append(x)
            fp_values.append(fp_map[x]/fp_count_map[x])
        #     print(fp_map[x]/fp_count_map[x])
                
#        print(fp_values)
        tp_values = []
        for x in tp_count_map.keys():
            tp_values.append(tp_map[x]/tp_count_map[x])
#            print(tp_map[x]/tp_count_map[x])
                
#        print(tp_values)
        
        plt.rcParams["figure.figsize"] = (20, 10)
        plt.rcParams.update({'font.size': 32})
        x = np.arange(len(x_values))
        
        fig, ax1 = plt.subplots()
        
        color = 'tab:red'
        ax1.set_xlabel('error threshold')
        ax1.set_ylabel('FPR', color=color)
        ax1.plot(x, fp_values, marker='o', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_values, rotation=20)
        ax1.grid()
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        color = 'tab:blue'
        ax2.set_ylabel('recall', color=color)  # we already handled the x-label with ax1
        ax2.plot(x, tp_values, marker='^', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        # ax2.grid()
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.savefig("../simulation_result/with_one_month/"+model_name+".png")
        plt.show()















#parent_folder_name = "../simulation_result/"
#folder_list=glob.glob(parent_folder_name+"*")
##print(folder_list)
#
#all_value_list=[]
#count=0
#for x in folder_list:
#    if "ST12000NM0007" in x:
#        with open(x,"r+") as in_file:
#            for line in in_file:
#                if len(line.strip())>0:
##                    count+=1
#                    if "Threshold" in line:
#                        all_value_list.append(line)
#                    if "TP, FP, TN, FN" in line:
#                        all_value_list.append(line)
#                
#print(len(all_value_list))
#print(count)
#
#for x in range(len(all_value_list)):
#    if "Threshold" in all_value_list[x]:
#        parts = all_value_list[x].split(" ")
#        print(parts[1])






#
#index = line.rfind("=")
#values = line[index+1:].strip()
#parts = values.split(" ")
#if int(float(parts[0]))!=0:
#    print(parts)









#    break
          
#date_list=["2019-07-01", "2019-07-02", "2019-07-03", "2019-07-04", "2019-07-05", "2019-07-06", "2019-07-07", "2019-07-08","2019-07-09", "2019-07-10","2019-07-11","2019-07-12","2019-07-13","2019-07-14","2019-07-15"]          
#prev = 0.0
#count=0
#serial_list=[]
#with open("/home/masudulhasanmasudb/eclipse-workspace/Proactive_error_detection/map_2019.txt","r+") as in_file:
#    for line in in_file:
#        if len(line.strip())>0:
#            if ".csv" not in line:
#                parts = line.split(" ")
#                value = float(parts[1].strip())
#                if value!=prev:
#                    if parts[0].strip() in date_list:
#                        print(model_name+" "+line)
#                        if model_name not in serial_list:
#                            serial_list.append(model_name)
#                        count+=1
#                prev = value
#            else:
#                model_name = line.strip()
#                
#                
#print(count)
#print(len(serial_list))
#print(serial_list)
#
#disk_model_map={}
#
#selected_disk=[]
#
##selected_models = ['ST4000DM000', 'ST8000DM002', 'ST12000NM0007', 'ST8000NM0055', 'ST3000DM001', 'ST4000DX000']
#selected_models = ['ST12000NM0007']
#for x in serial_list:
#    with open("/home/masudulhasanmasudb/eclipse-workspace/Proactive_error_detection/output_files_2019/"+x,"r+") as in_file:
#        for line in in_file:
#            print(line)
#            parts=line.split(",")
##            if parts[2].strip() not in disk_model_map:
##                disk_model_map[parts[2].strip()]=1
##            else:
##                value = disk_model_map[parts[2].strip()]
##                disk_model_map[parts[2].strip()]=value+1
#            if parts[2].strip() in selected_models:
#                index = x.find(".")
#                selected_disk.append(x[:index])
#            break
#                
#                
#print(len(selected_disk))            
#print(selected_disk)
                