#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:26:43 2020

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


disk_model=["ST8000DM002", "ST8000NM0055","ST4000DX000","ST4000DM000","ST12000NM0007","ST3000DM001"]
model_name_list=[]
accuracy_list=[]
recall_list=[]
with open("/home/masudulhasanmasudb/Downloads/rf.txt","r+")as in_file:
    for line in in_file:
        if "model" in line:
            colon_index=line.rfind(":")
            model_name = line[colon_index+1:].strip()
            model_name_list.append(model_name)
        if "Accuracy" in line:
            colon_index=line.rfind(":")
            value = float(line[colon_index+1:].strip())
            accuracy_list.append(value)
        if "Recall" in line:
            colon_index=line.rfind(":")
            value = float(line[colon_index+1:].strip())
            recall_list.append(value)

print(model_name_list)

model_dict={}
for x in range(len(model_name_list)):
    if model_name_list[x] in disk_model:
        model_dict[model_name_list[x]]=recall_list[x]
#        x_list.append(model_name_list[x])
#        y_list.append(recall_list[x])

x_list=[]
y_list=[]

count=0
total_value=0
for x in sorted(model_dict.items(), key = lambda kv:(kv[1], kv[0])):
    print(x)
    x_list.append(x[0])
    y_list.append(x[1]*100)
    total_value+=x[1]

average = float(total_value/len(x_list))*100
print(average)

#for x in range(len(model_name_list)):
#    if model_name_list[x] in disk_model:
#        x_list.append(model_name_list[x])
#        y_list.append(recall_list[x])
    

plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams.update({'font.size': 32})
x = np.arange(len(x_list))
fig, ax = plt.subplots()
ax.plot(x_list,y_list, marker='o', label="Actual Recall value")
ax.axhline(y=average, xmin=0.0, xmax=1.0, color='r', label="Average Reacll value")
ax.set(ylabel='Recall', xlabel='Disk Model')
ax.set_xticks(x)
ax.set_xticklabels(x_list, rotation=20)
ax.grid()
ax.legend()
fig.tight_layout()
plt.savefig("../recall_vallues_with_undersampling.pdf")
plt.show()

#parent_folder_name = "/home/masudulhasanmasudb/Downloads/transfer_learning_result-20200118T015333Z-001/transfer_learning_result/"
#folder_list=glob.glob(parent_folder_name+"*")
##print(folder_list)
#
#for year in range(2016,2020):
#    model_name_list=[]
#    accuracy_list=[]
#    recall_list=[]
#    for x in folder_list:
#        if "png" not in x and "year" not in x and str(year) in x:
#            print(x)
#            with open(x,"r+") as in_file:
#                for line in in_file:
#                    if "model" in line:
#                        colon_index=line.rfind(":")
#                        model_name = line[colon_index+1:].strip()
#                        model_name_list.append(model_name)
#                    if "Accuracy" in line:
#                        colon_index=line.rfind(":")
#                        value = float(line[colon_index+1:].strip())
#                        accuracy_list.append(value)
#                    if "Recall" in line:
#                        colon_index=line.rfind(":")
#                        value = float(line[colon_index+1:].strip())
#                        recall_list.append(value)
#    
#    plt.rcParams["figure.figsize"] = (20, 10)
#    plt.rcParams.update({'font.size': 24})
#    width = 0.35  # the width of the bars
#    x = np.arange(len(model_name_list))
#    fig, ax = plt.subplots()
#    rects1 = ax.bar(x - width/2, accuracy_list, width, label='Accuracy')
#    rects2 = ax.bar(x + width/2, recall_list, width, label='Recall')
#    
#    # Add some text for labels, title and custom x-axis tick labels, etc.
#    ax.set_ylabel('Scores')
#    ax.set_title("Accuracy for different Disk Models "+str(year))
#    ax.set_xticks(x)
#    ax.set_xticklabels(model_name_list, rotation = 45)
#    ax.legend()
#    fig.tight_layout()
#    plt.savefig("/home/masudulhasanmasudb/Downloads/transfer_learning_result-20200118T015333Z-001/transfer_learning_result/"+str(year)+".png")
#    plt.show()

#value_list=[]
#with open("/home/masudulhasanmasudb/eclipse-workspace/Proactive_error_detection/day_wise_stat.txt","r+") as in_file:
#    for line in in_file:
##        print(line)
#        value_list.append(int(line.strip()))
#
#plt.rcParams["figure.figsize"] = (20, 10)
#plt.rcParams.update({'font.size': 24})
#x = np.arange(len(value_list))
#fig, ax = plt.subplots()
#ax.plot(x, value_list)
#
#ax.set(xlabel='time (s)', ylabel='#of daily disk failure', title='Daily Disk Failure From 2013-2019')
#ax.grid()
#
#fig.savefig("../day_wise.png")
#plt.show()

#bit_error_map={}
#with open("../combined_file/bit_error_result.txt","r+") as in_file:
#    for line in in_file:
##        print(line)
#        colon_index=line.rfind(":")
#        key = line[:colon_index].strip()
#        value = line[colon_index+1:].strip()
#        bit_error_map[key]=int(value)
#
#
##print(bit_error_map)
#
#now = datetime.datetime(2013,4,9)
#value_list=[]
#while True: 
#    now += datetime.timedelta(days=1)
#    if str(now.date())=="2019-09-30":
#        break
##    print(now.date())
##    print(bit_error_map[str(now.date())])
#    try:
#        value_list.append(int(bit_error_map[str(now.date())]))
#    except:
#        print(now)
#    
#plt.rcParams["figure.figsize"] = (20, 10)
#plt.rcParams.update({'font.size': 24})
#x = np.arange(len(value_list))
#fig, ax = plt.subplots()
#ax.plot(x, value_list)
#
#ax.set(xlabel='time (s)', ylabel='#of daily bit error', title='Daily Bit Error From 2013-2019')
#ax.grid()
#
#fig.savefig("../day_wise_bit_error.png")
#plt.show()

#text_list=[]
#with open("../iv_time.txt","r+")as in_file:
#    for line in in_file:
#        print(line)
#        text_list.append(line)
#
#file_size_list=[]
#in_memory =[]
#in_disk=[]
#for x in range(len(text_list)):
#    if "file_size" in text_list[x]:
#        index = text_list[x].rfind("=")
#        file_size_list.append(text_list[x][index+1:].strip())
#        in_memory.append(float(text_list[x+1].strip()))
#        in_disk.append(float(text_list[x+2].strip()))
#
#print(in_memory)
#print(in_disk)
#
#x = np.arange(len(file_size_list))  # the label locations
#width = 0.35  # the width of the bars
#
#plt.rcParams["figure.figsize"] = (16, 8)
#plt.rcParams.update({'font.size': 24})
#
#fig, ax = plt.subplots()
#rects1 = ax.bar(x - width/2, in_memory, width, label='Memory')
#rects2 = ax.bar(x + width/2, in_disk, width, label='Disk')
#
## Add some text for labels, title and custom x-axis tick labels, etc.
#ax.set_ylabel('computation time(log scale)')
#ax.set_xlabel('file size')
##ax.set_title('Cheksum computation time comparision between memory and disk')
#ax.set_xticks(x)
#ax.set_xticklabels(file_size_list)
#ax.legend()
#plt.yscale('log')
#
#fig.tight_layout()
#plt.savefig("../memory_disk_time_comparision.png")
#plt.show()



#bit_error_map={}
#with open("../bit_error_result.txt","r+") as in_file:
#    for line in in_file:
##        print(line)
#        colon_index=line.rfind(":")
#        key = line[:colon_index].strip()
#        value = line[colon_index+1:].strip()
#        bit_error_map[key]=int(value)
#
#
##print(bit_error_map)
#
#now = datetime.datetime(2013,4,9)
#bit_value_list=[]
#while True: 
#    now += datetime.timedelta(days=1)
#    if str(now.date())=="2019-09-30":
#        break
##    print(now.date())
##    print(bit_error_map[str(now.date())])
#    try:
#        bit_value_list.append(int(bit_error_map[str(now.date())]))
#    except:
#        print(now)
#
#
#
#
#sector_error_map={}
#with open("../sector_error_result.txt","r+") as in_file:
#    for line in in_file:
##        print(line)
#        colon_index=line.rfind(":")
#        key = line[:colon_index].strip()
#        value = line[colon_index+1:].strip()
#        sector_error_map[key]=int(value)
#
#
##print(bit_error_map)
#
#now = datetime.datetime(2013,4,9)
#sector_value_list=[]
#while True: 
#    now += datetime.timedelta(days=1)
#    if str(now.date())=="2019-09-30":
#        break
##    print(now.date())
##    print(bit_error_map[str(now.date())])
#    try:
#        sector_value_list.append(int(sector_error_map[str(now.date())]))
#    except:
#        print(now)
#    
#
#    
#each_day_file_list =[]
#with open("../number_of_disk_each_day.txt","r+") as in_file:
#    for line in in_file:
#        each_day_file_list.append(int(line.strip()))
#        
#print(len(each_day_file_list))
#for year in range(2013,2020):
#    for month in range(1,13):
#        for day in range(1,32):
#            month_str=str(month)
#            if month<=9:
#                month_str="0"+str(month)
#            
#            day_str=str(day)
#            if day<=9:
#                day_str="0"+str(day)
#            
#            try:
#                x = "../modified_data/"+str(year)+"-"+str(month_str)+"-"+str(day_str)+".csv"
#                df = pd.read_csv(x, header=None)
##                print(len(df))
#                each_day_file_list.append(int(len(df)))
#            except:
#                print(x)
#                print("file not found")
#out_file = open("number_of_disk_each_day.txt","a+")            
#for x in (each_day_file_list):
#    out_file.write(str(x)+"\n")
#    out_file.flush()

#bit_error_ratio_list=[] 
#for x in range(len(bit_value_list)):
#    bit_error_ratio_list.append(float(bit_value_list[x]/each_day_file_list[x]))
#sector_error_ratio_list=[] 
#for x in range(len(sector_value_list)):
#    sector_error_ratio_list.append(float(sector_value_list[x]/each_day_file_list[x]))
#
#
#plt.rcParams["figure.figsize"] = (20, 10)
#plt.rcParams.update({'font.size': 24})
#
#x = np.arange(len(bit_error_ratio_list))
#fig, ax = plt.subplots()
#ax.plot(x, bit_error_ratio_list)
#
#ax.set(xlabel='time (s)', ylabel='#ratio of daily bit error', title='Ratio of Bit Error From 2013-2019')
#ax.grid()
#
#fig.savefig("../bit_error_ratio.png")
#plt.show()
#
#x = np.arange(len(sector_error_ratio_list))
#fig, ax = plt.subplots()
#ax.plot(x, sector_error_ratio_list)
#
#ax.set(xlabel='time (s)', ylabel='#ratio of daily sector error', title='Ratio of Sector Error From 2013-2019')
#ax.grid()
#
#fig.savefig("../sector_error_ratio.png")
#plt.show()


#fig, ax1 = plt.subplots()
#
#color = 'tab:red'
#ax1.set_xlabel('time (day)')
#ax1.set_ylabel('#of daily sector error', color=color)
#x = np.arange(len(sector_value_list))
#
#ax1.plot(x, sector_value_list, color=color)
#ax1.tick_params(axis='y', labelcolor=color)
#
#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
#color = 'tab:blue'
#ax2.set_ylabel('#of disk in each day', color=color)  # we already handled the x-label with ax1
#x = np.arange(len(each_day_file_list))
#ax2.plot(x, each_day_file_list, color=color)
#ax2.tick_params(axis='y', labelcolor=color)
#
#fig.tight_layout()  # otherwise the right y-label is slightly clipped
#fig.savefig("../sector_error_ratio.png")
#plt.show()
#
#
#
#fig, ax1 = plt.subplots()
#
#color = 'tab:red'
#ax1.set_xlabel('time (day)')
#ax1.set_ylabel('#of daily bit error', color=color)
#x = np.arange(len(bit_value_list))
#
#ax1.plot(x, bit_value_list, color=color)
#ax1.tick_params(axis='y', labelcolor=color)
#
#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
#color = 'tab:blue'
#ax2.set_ylabel('#of disk in each day', color=color)  # we already handled the x-label with ax1
#x = np.arange(len(each_day_file_list))
#ax2.plot(x, each_day_file_list, color=color)
#ax2.tick_params(axis='y', labelcolor=color)
#
#fig.tight_layout()  # otherwise the right y-label is slightly clipped
#fig.savefig("../bit_error_ratio.png")
#plt.show()
























#fpr_lists=[]
#
#with open("/home/masudulhasanmasudb/Downloads/sector_error_fpr.csv","r+") as in_file:
#    for line in in_file:
##        print(line)
#        parts=line.strip().split(",")[:-1]
##        print(parts)
#        new_list=[]
#        for item in parts:
#            try:
#                new_list.append(float(item))
#            except:
#                print("error "+item)
##        parts = [float(item) for item in parts]
##        print(len(parts))
#        fpr_lists.append(new_list)
#        
#recall_list=[]
#with open("/home/masudulhasanmasudb/Downloads/sector_error_recall.csv","r+") as in_file:
#    for line in in_file:
##        print(line)
#        parts=line.strip().split(",")
#        new_list=[]
#        for item in parts:
#            try:
#                new_list.append(float(item))
#            except:
#                print("recall error "+item)
#        recall_list.append(new_list)
#
##print(recall_list)
#threshold_list = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]
#print(len(threshold_list))
#selected_models = ['ST4000DM000', 'ST8000DM002', 'ST12000NM0007', 'ST8000NM0055', 'ST3000DM001', 'ST4000DX000']
#for x in range(len(selected_models)):
#    model_name = selected_models[x]
#    plt.rcParams["figure.figsize"] = (20, 10)
#    plt.rcParams.update({'font.size': 24})
#    
#    X_list = threshold_list
#    fpr_y = fpr_lists[x][:15]
#    recall_y = recall_list[x][:15]
#    
#    threshold_values =[y for y in range(15)]
#    
#    fig, axs = plt.subplots(2)
#    fig.suptitle(model_name)
#    axs[0].plot(threshold_values, fpr_y, 'o-')
#    axs[0].set_ylabel('FPR')
#    axs[0].set_xticks(threshold_values)
#    axs[0].set_xticklabels(X_list, fontsize=24)
#    axs[0].grid()
#    
#    axs[1].plot(threshold_values, recall_y, '.-')
#    axs[1].set_xlabel("Threshold")
#    axs[1].set_ylabel('Recall Value')
#    axs[1].set_xticks(threshold_values)
#    axs[1].set_xticklabels(X_list, fontsize=24)
#    axs[1].grid()
#    plt.savefig("../figures_for_different_threshold/sector_error/"+str(x+1)+".png")



#selected_models = ['ST8000DM002', 'ST4000DX000', 'ST12000NM0007', 'ST4000DM000','ST8000NM0055','ST3000DM001']
#for model_name in selected_models:
#    has_already_started = False
#    threshold =0.5
#    recall_list=[]
#    fpr_list=[]
#    with open("/home/masudulhasanmasudb/Downloads/result_1.txt","r+") as in_file:
#        for line in in_file:
#            if "Model" in line and has_already_started:
#                break
#            if model_name in line and not has_already_started:
#                has_already_started = True
#            if has_already_started:
#                if "Threshold" in line:
#                    parts = line.split(" ")
#                    threshold=float(parts[1].strip())
#                if "Recall" in line:
#                    colon_index = line.find(":")
#                    recall_value = float(line[colon_index+1:])
#                    recall_list.append((threshold,recall_value))
#                if "extra" in line:
#                    colon_index = line.find(":")
#                    extra_value = float(line[colon_index+1:])
#                    fpr_list.append((threshold,extra_value))
#        print(model_name)
#        
#        fpr_y=[]
#        recall_y=[]
#        X_list=[]
#        for (x,y) in fpr_list:
#            X_list.append(x)
#            fpr_y.append(y)
#        
#        for (x,y) in recall_list:
#            recall_y.append(y)
#        
#        X_list = ["%.2f"%item for item in X_list]
#        
#        plt.rcParams["figure.figsize"] = (16, 10)
#        plt.rcParams.update({'font.size': 24})
#        
#        fig, axs = plt.subplots(2)
#        fig.suptitle(model_name)
#        axs[0].plot(X_list, fpr_y, 'o-')
#        axs[0].set_ylabel('FPR')
#        axs[0].grid()
#        
#        axs[1].plot(X_list, recall_y, '.-')
#        axs[1].set_xlabel("Threshold")
#        axs[1].set_ylabel('Recall Value')
#        axs[1].set_xticks(X_list)
#        axs[1].grid()
#        
#        
##        fig =plt.subplot(2, 1, 1)
##        fig.plot(X_list, fpr_y, 'o-')
##        plt.title(model_name)
##        fig.set_ylabel('FPR')
##        
##        fig = plt.subplot(2, 1, 2)
##        fig.plot(X_list, recall_y, '.-')
##        fig.set_xlabel("Threshold")
##        fig.set_ylabel('Recall Value')
##        fig.set_xticks(X_list)
##        plt.show()
#        plt.savefig("../figures_for_different_threshold/"+model_name+".png")
            
        