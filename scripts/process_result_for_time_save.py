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

parent_folder_name = "../simulation_result/with_six_months/"
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
        
        ts_map={}
        ts_count_map={}

        for x in range(len(all_value_list)):
            if "Threshold" in all_value_list[x]:
#                print(all_value_list[x+1])
                parts = all_value_list[x].split(" ")
                threshold_value = float(parts[1].strip())

                value_parts = all_value_list[x+1].split(" ")
                tp = int(value_parts[5].strip())
                fp = int(value_parts[6].strip())
                tn = int(value_parts[7].strip())
                fn = int(value_parts[8].strip())
                
                if tp+fp+tn+fn!=0:
                    time_save = (tn+fn)/(tp+fp+tn+fn)
                    if threshold_value not in ts_map:
                        ts_map[threshold_value] = time_save
                    else:
                        prev_value = ts_map[threshold_value]
                        ts_map[threshold_value] = prev_value + time_save
                        
                    if threshold_value not in ts_count_map:
                        ts_count_map[threshold_value] = 1
                    else:
                        prev_value = ts_count_map[threshold_value]
                        ts_count_map[threshold_value] = prev_value + 1
                        
        
        x_values = []
        ts_values = []
        for x in ts_count_map.keys():
            x_values.append(x)
            ts_values.append((ts_map[x]/ts_count_map[x])*100)
                


        plt.rcParams["figure.figsize"] = (20, 10)
        plt.rcParams.update({'font.size': 32})
        x = np.arange(len(x_values))
        
        fig, ax1 = plt.subplots()
        
        color = 'tab:blue'
        ax1.set_xlabel('error threshold')
        ax1.set_ylabel('Time saved(%)')
        ax1.plot(x, ts_values, marker='o', color=color)
        ax1.tick_params(axis='y')
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_values, rotation=20)
        ax1.grid()
        
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.savefig("../simulation_result/with_six_months/"+model_name+"_ts.png")
        plt.show()


