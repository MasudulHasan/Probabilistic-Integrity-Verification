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
parent_folder_name = "/home/masudulhasanmasudb/eclipse-workspace/Proactive_error_detection/output_files/"
#folder_list=glob.glob(parent_folder_name+"*")
##out_file = open("done_so_far.txt","a+")
#
#
#
features = [1, 4, 5, 7, 9, 12, 187, 188, 193, 194, 196, 197, 198, 199]
columns_specified = []
for feature in features:
    	columns_specified += ["smart_{0}_raw".format(feature)]
#columns_specified = ["serial_number", "date", "model", "failure"] + columns_specified
#columns_specified = ["model"] + columns_specified
#print(columns_specified)
sample_data = pd.DataFrame(columns=columns_specified)
#print(sample_data)        
#
#count = 0
#hdd_list = []
#for x in folder_list:
##    with open(x,"r") as in_file:
##        for line in in_file:
#    df = pd.read_csv(x)
#    df = df[columns_specified]
##    print(df.columns)
#    print(df['model'][0])
#    count+=1
#    hdd_list.append(df['model'][0])
#
#print(collections.Counter(hdd_list))
   
#    if count==10:
#        break






#file_list=[]

#for x in folder_list:
#    count = 0
#    with open(x,"r") as in_file:
#        for line in in_file:
#            parts = line.split(",")
#            count+=1
#            year_index = parts[0].find("-")
#            year = int(parts[0][:year_index])
#            print(year)
#            
#            if "ST4000DM000" in parts[2] and year>=2016:
#                file_list.append(x)
#                break
#            if count==1:
#                break
#
#print(file_list)
#for x in file_list:
#    if "Z30330GD" in x:
#         with open(x,"r") as in_file:
#            for line in in_file:
#                parts = line.split(",")
#                print(str(parts[0])+" "+ \
#                      str(parts[1])+" "+ \
#                      str(parts[2])+" "+ \
#                      str("Read Error Rate "+parts[6])+ " "+ \
#                      str("Start/Stop Count "+parts[12])+ " "+ \
#                      str("Reallocated Sectors Count "+parts[14])+ " "+\
#                      str("Seek Error Rate "+parts[16])+ " "+\
#                      str("Power-On Hours "+parts[20])+" "+\
#                      str("Power Cycle Count "+parts[26])+" "+\
#                      str("Reported Uncorrectable Errors "+parts[62])+" "+\
#                      str("Load Cycle Count "+parts[74])+" "+\
#                      str("Temperature "+parts[76])+" "+\
#                      str("Reallocation Event Count "+parts[80])+" "+\
#                      str("Current Pending Sector Count "+parts[82])+" "+\
#                      str("Uncorrectable Sector Count "+parts[84])+" "+\
#                      str("UltraDMA CRC Error Count "+parts[86]))
#                    
 

#count = 0
file_list=[]
#with open("model_files.txt","r")as in_file:
#        for line in in_file:
##            print(line)
#            file_list.append(line)
##            parts = line.split(",")
##            print(parts)
##            parameter_list=[x for x in parts]
##            print(parameter_list)
##            count+=1
##            if(count==1):
##                break
#
count = 0
#files_with_error = []
#output_file = open("files_with_error.txt","a+")
#for x in file_list:
#    df = pd.read_csv(parent_folder_name+str(x).strip())
#    df = df[columns_specified]
##    print(df.columns)
##    df_list = df["smart_187_raw"]
##    print(df["smart_187_raw"])
#    if(not df["smart_187_raw"].nunique()==1):
#        files_with_error.append(x)
#        print(df["smart_187_raw"].nunique()==1)
##    print(df["smart_198_raw"].value_counts())
#    count+=1
##    if(count==500):
##        break
##    hdd_list.append(df['model'][0])
#
#for x in  files_with_error:
#    output_file.write(x+"\n")
#    output_file.flush()



file_list=[]
with open("files_with_error.txt","r")as in_file:
        for line in in_file:
#            print(line)
            if(len(line.strip())>0):
                file_list.append(line)
                count+=1
#            if(count==1):
#                break
print(count)

test_size =int( 0.1*count)
print(int(test_size))
train_size = count-test_size


for x in range(train_size):
    print(x)
    
    df = pd.read_csv(parent_folder_name+str(file_list[x]).strip())
    df = df[columns_specified]
#    print(df)
    df_list = df["smart_187_raw"].tolist()
    
#    print(df["smart_187_raw"])
#    print(type(df_list))
#    print(df_list)
#    if(not df["smart_187_raw"].nunique()==1):
#        files_with_error.append(x)
#        print(df["smart_187_raw"].nunique()==1)
#    print(df["smart_198_raw"].value_counts())
    label_list=[]
    for i in range(len(df_list)-1):
        if df_list[i+1]!= df_list[i]:
            label_list.append(1)
        else:
           label_list.append(0) 
    
#    print(len(label_list))
#    print(len(df_list))
    
    last_row = len(df)-1
    df = df[:-1]
    df['label'] = label_list
#    print(df)
    
    df.to_csv("data_files/"+file_list[x],index=False)
    
#    count+=1
#    if(count==50):
#        break
#    hdd_list.append(df['model'][0])

out_file= open("files_for_test.txt","a+")
for x in range(train_size,count):
    out_file.write(str(x)+"\n")
    out_file.flush()





#for x in file_list:
#    print(x)
#    df = pd.read_csv(parent_folder_name+str(x).strip())
#    df = df[columns_specified]
##    print(df)
#    df_list = df["smart_187_raw"].tolist()
#    
##    print(df["smart_187_raw"])
##    print(type(df_list))
##    print(df_list)
##    if(not df["smart_187_raw"].nunique()==1):
##        files_with_error.append(x)
##        print(df["smart_187_raw"].nunique()==1)
##    print(df["smart_198_raw"].value_counts())
#    label_list=[]
#    for i in range(len(df_list)-1):
#        if df_list[i+1]!= df_list[i]:
#            label_list.append(1)
#        else:
#           label_list.append(0) 
#    
##    print(len(label_list))
##    print(len(df_list))
#    
#    last_row = len(df)-1
#    df = df[:-1]
#    df['label'] = label_list
##    print(df)
#    
#    df.to_csv("data_set/"+x,index=False)
#    
##    count+=1
##    if(count==50):
##        break
##    hdd_list.append(df['model'][0])
#
#    
#    
#
##for i in range(len(parameter_list)):
##    print(str(parameter_list[i]).strip()+" -> "+str(i))            
#    
