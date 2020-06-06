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

parent_folder_name = "../simulation_result/"
folder_list=glob.glob(parent_folder_name+"*")
disk_model=["ST8000DM002", "ST8000NM0055","ST4000DM000","ST12000NM0007"]
for disk_model_name in disk_model:
    
    all_fp_list=[]
    all_tp_list=[]
    X_list=[]
    legend_list=[]
    
    for folder in folder_list:
#        print(folder)
        s_index = folder.rfind("with_")
        e_index= folder.rfind("_m")
        
        if e_index==-1:
            legend_list.append("all")
        else:
            window_size = folder[s_index+5:e_index]
            legend_list.append(window_size)
        
        file_list=glob.glob(folder+"/*")       
        for x in file_list:
            if disk_model_name in x and ".png" not in x:
                all_value_list=[]
                start_index = x.rfind("/")
                end_index = x.rfind(".txt")
                model_name = x[start_index+1:end_index]
                print(model_name)
                with open(x,"r+") as in_file:
                    for line in in_file:
                        if len(line.strip())>0:
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
                    
                all_fp_list.append(fp_values)
                X_list = x_values
    
                tp_values = []
                for x in tp_count_map.keys():
                    tp_values.append(tp_map[x]/tp_count_map[x])
                
                all_tp_list.append(tp_values)
            
    print(legend_list)
    print(len(all_tp_list))

            
    plt.rcParams["figure.figsize"] = (20, 10)
    plt.rcParams.update({'font.size': 32})
    x = np.arange(len(X_list))
    
    fig, ax = plt.subplots()
    marker_list = ['^', 'o', '*', 'v', '<']
    for y in range(len(all_tp_list)):
        ax.set_xlabel('error threshold')
        ax.set_ylabel('recall')
        ax.plot(x, all_tp_list[y], marker=marker_list[y], label=legend_list[y])
        ax.tick_params(axis='y')
        ax.set_xticks(x)
        ax.set_xticklabels(x_values, rotation=20)
        ax.grid()
    
    ax.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig("../combined_figures/"+model_name+".png")
    plt.show()
    
    
    fig, ax = plt.subplots()
    marker_list = ['^', 'o', '*', 'v', '<']
    for y in range(len(all_fp_list)):
        ax.set_xlabel('error threshold')
        ax.set_ylabel('FPR')
        ax.plot(x, all_fp_list[y], marker=marker_list[y], label=legend_list[y])
        ax.tick_params(axis='y')
        ax.set_xticks(x)
        ax.set_xticklabels(x_values, rotation=20)
        ax.grid()
    
    ax.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig("../combined_figures/"+model_name+"_fp.png")
#    plt.show()
    
    
#    fig, ax1 = plt.subplots()
#    
#    color = 'tab:red'
#    ax1.set_xlabel('error threshold')
#    ax1.set_ylabel('FPR', color=color)
#    ax1.plot(x, fp_values, marker='o', color=color)
#    ax1.tick_params(axis='y', labelcolor=color)
#    ax1.set_xticks(x)
#    ax1.set_xticklabels(x_values, rotation=20)
#    ax1.grid()
#    
#    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#    
#    color = 'tab:blue'
#    ax2.set_ylabel('recall', color=color)  # we already handled the x-label with ax1
#    ax2.plot(x, tp_values, marker='^', color=color)
#    ax2.tick_params(axis='y', labelcolor=color)
#    # ax2.grid()
#    fig.tight_layout()  # otherwise the right y-label is slightly clipped
#    fig.savefig("../simulation_result/with_one_month/"+model_name+".png")
#    plt.show()

                