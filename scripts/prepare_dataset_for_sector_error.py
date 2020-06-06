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
import gc,sys, traceback
import pandas as pd
import collections

#day = int(sys.argv[1])
day =1
month = 6
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

#parent_folder_name = "/home/masudulhasanmasudb/eclipse-workspace/Proactive_error_detection/disk_model_files/"
parent_folder_name = "/home/masudulhasanmasudb/eclipse-workspace/Proactive_error_detection/output_files_2019/"
folder_list=glob.glob(parent_folder_name+"*")
#print(folder_list)

features = [1, 4, 5, 7, 9, 12, 187, 188, 193, 194, 196, 197, 198, 199]
columns_specified = []
for feature in features:
    	columns_specified += ["smart_{0}_raw".format(feature)]
columns_specified = ["serial_number", "date", "model", "failure"] + columns_specified
#columns_specified = ["date","model","failure"] + columns_specified
print(columns_specified)
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
            # df = df[(df[1] > '2019-05-31') & (df[1] < '2019-07-02')]
            df = df.reset_index()
#            print(df)
            if df.shape[0]>1:
            
#            df = pd.read_csv(x)
#            df = df[columns_specified]
#                print(df)
                model_name = df[2][1]
                print(model_name)
                if model_name != "00MD00":
                    df_list = df[6].tolist()
                    label_list= get_label_list(df_list, day)
                    
                    bit_error_list = df[10].tolist()
                    bit_error_labels= get_label_list(bit_error_list, day)
                    
                    failure_list = df[3].tolist()
                    
                    df = df.drop([0,1,2,3], axis=1)
                    last_row = len(df)-1
                    
                    diff = len(df_list) - len(label_list)
                    
                    df = df[:-diff]
                    df['bit_error_label'] = bit_error_labels
                    df['sector_error_label'] = label_list
                    df['failure'] = failure_list[:len(df)]
    #                print(df)
                    output_file = open("../one_month_files/"+str(month)+"/"+model_name+".csv","a+")
                    df.to_csv(output_file, header=False, index=False)
                
                    count+=1
                    print(count)
    #                if(count==3):
    #                    break
#            del df
#            gc.collect()
        except:
            print(x)
            traceback.print_exc()