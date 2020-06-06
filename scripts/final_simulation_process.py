#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 21:17:49 2020

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

def calculate_accuracy(predicted_value, prob_list, real_values):
    threshold_list = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]
    
    final_string=""
    
    fp_list=[]
    tp_list=[]
    
    for base_threshold in threshold_list:
        tp=0 
        tn=0
        fp=0
        fn=0
        for x in range(len(prob_list)):
            prob = float(prob_list[x])
            if prob >= base_threshold:
                if real_values[x]==1:
                  tp+=1
                else:
                  fp+=1
            else:
                if predicted_value[x]== 1 and real_values[x]==1:
                  tp+=1
                elif predicted_value[x]== 0 and real_values[x]==1:
                  fn+=1
                
                elif predicted_value[x]== 0 and real_values[x]==0:
                  tn+=1
                elif predicted_value[x]== 1 and real_values[x]==0:
                  fp+=1
                  
#        print("TP, FP, TN, FN = "+str(tp)+" "+str(fp)+" "+str(tn)+" "+str(fn))

        recall = (tp/(tp+fn))*100
        fpr = (fp/(tn+fp))*100
        fp_list.append(fpr)
        tp_list.append(recall)
        
        
    return tp_list, fp_list
#    plt.rcParams["figure.figsize"] = (20, 10)
#    plt.rcParams.update({'font.size': 32})
#    x = np.arange(len(threshold_list))
#    
#    fig, ax1 = plt.subplots() 
#    color = 'tab:red'
#    ax1.set_xlabel('error threshold')
#    ax1.set_ylabel('FPR', color=color)
#    ax1.plot(x, fp_list, marker='o', color=color)
#    ax1.tick_params(axis='y', labelcolor=color)
#    ax1.set_xticks(x)
#    ax1.set_xticklabels(threshold_list, rotation=20)
#    ax1.grid()
#    
#    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#    
#    color = 'tab:blue'
#    ax2.set_ylabel('recall', color=color)  # we already handled the x-label with ax1
#    ax2.plot(x, tp_list, marker='^', color=color)
#    ax2.tick_params(axis='y', labelcolor=color)
#    # ax2.grid()
#    fig.tight_layout()  # otherwise the right y-label is slightly clipped
##    fig.savefig("../simulation_result/with_one_month/"+model_name+".png")
#    plt.show()
        
        
model_name = "ST8000DM002"
parent_folder_name = "../final_simulation_logs/"
folder_list=glob.glob(parent_folder_name+"*")
#print(folder_list)

tpr_lists=[]
fpr_lists=[]
legend_list=[]
for file in folder_list:
    print(file)
    s_index=file.rfind("/")
    e_index=file.rfind(".txt")
    
    file_name=file[s_index+1:e_index]
    
    predicted_values=[]
    real_values =[]
    prob_values=[]
    with open(file,"r+") as in_file:
        for line in in_file:
#            print(line)
            parts=line.split(" ")
#            print(parts)
            predicted_values.append(int(parts[0].strip()))
            real_values.append(int(parts[1].strip()))
            prob_values.append(float(parts[2].strip()))
        
    tp_list, fp_list = calculate_accuracy(predicted_values, prob_values, real_values)
    tpr_lists.append(tp_list)
    fpr_lists.append(fp_list)
    legend_list.append(file_name)
    
threshold_list = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]
plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams.update({'font.size': 32})
x = np.arange(len(threshold_list))

fig, ax = plt.subplots()
marker_list = ['^', 'o', '*', 'v', '<','>']
for y in range(len(tpr_lists)):
    ax.set_xlabel('error threshold')
    ax.set_ylabel('recall')
    ax.plot(x, tpr_lists[y], marker=marker_list[y], label=legend_list[y])
    ax.tick_params(axis='y')
    ax.set_xticks(x)
    ax.set_xticklabels(threshold_list, rotation=20)
    ax.grid(True)

ax.legend()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.savefig("../combined_figures/"+model_name+"_combined.png")
plt.show()


fig, ax = plt.subplots()
#marker_list = ['^', 'o', '*', 'v', '<']
for y in range(len(fpr_lists)):
    ax.set_xlabel('error threshold')
    ax.set_ylabel('FPR')
    ax.plot(x, fpr_lists[y], marker=marker_list[y], label=legend_list[y])
    ax.tick_params(axis='y')
    ax.set_xticks(x)
    ax.set_xticklabels(threshold_list, rotation=20)
    ax.grid(True)

ax.legend()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.savefig("../combined_figures/"+model_name+"_combined_fp.png")
plt.show()
