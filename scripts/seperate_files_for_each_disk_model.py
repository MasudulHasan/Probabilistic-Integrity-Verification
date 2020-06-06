#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 11:56:22 2019

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

#disk_models =[]
#with open("/home/masudulhasanmasudb/eclipse-workspace/Proactive_error_detection/disk_model_count.txt") as in_file:
#    for line in in_file:
##        print(line)
#        index = line.find(":")
#        end_index = line.find("value:")
##        print(line[index+1:end_index].strip())
#        model_name = line[index+1:end_index].strip()
#        if model_name != "00MD00":
#            disk_models.append(model_name)


#disk_model = "ST8000NM0055"
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

parent_folder_name = "/home/masudulhasanmasudb/eclipse-workspace/Proactive_error_detection/hdd_disk_model_files/"
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
for day in range(7,8):
    for x in folder_list:
#        print(x)
        
        df = pd.read_csv(x)
        df = df[columns_specified]
    #    print(df)
        model_name = df["model"][0]
        print(model_name)
        if model_name != "00MD00":
#                print(x)
            df_list = df["smart_187_raw"].tolist()
            label_list= get_label_list(df_list, day)
#            for i in range(len(df_list)-1):
#                
#                if df_list[i+1]!= df_list[i]:
#                    label_list.append(1)
#                else:
#                   label_list.append(0) 
            
#            print(len(label_list))
#            print(len(df_list))
            
            df = df.drop(['model'], axis=1)
            last_row = len(df)-1
            
            diff = len(df_list) - len(label_list)
            
            df = df[:-diff]
            df['label'] = label_list
#            print(df)
            output_file = open("../dataset_2017/"+str(day)+"/"+model_name+".csv","a+")
            df.to_csv(output_file, header=False, index=False)
        
            count+=1
            print(count)
#            if(count==1):
#                break
        del df
        gc.collect()