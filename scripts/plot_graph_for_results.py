#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 13:20:21 2019

@author: masudulhasanmasudb
"""
import time
import glob,random
import datetime
import os
import subprocess
import shlex
import gc, traceback
import pandas as pd
import collections
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import ensemble, metrics 
import gc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from imblearn.over_sampling import (RandomOverSampler, SMOTE, ADASYN)
from collections import Counter

parent_folder_name = "/home/masudulhasanmasudb/Music/hdd_data/result_2017/"
folder_list=glob.glob(parent_folder_name+"*")
#print(folder_list)
for folder in folder_list:
#    if "/2" in folder:
        start_index = folder.find("_2017")
        day_number = folder[start_index+6:]
        print(day_number)
        file_list=glob.glob(folder+"/*")
    #    print(file_list)
        count=0
        fileOpen=0
        for file in file_list:
            if ".txt" in file and "svm" not in file:
#                print(file)
                with open(file) as in_file:
                    fileOpen+=1
                    for line in in_file:
                        if len(line.strip())>0:
                            if "model" in line:
                                count+=1
                if(fileOpen==1):
                    break
        print(count)
        resultList=[[] for i in range(count)]
        model_name_list=[]
#        print(resultList)
        disk_model_dict={}
        rev_dict={}
        k=0
        for file in file_list:
            if ".txt" in file and "svm" not in file and "knn" not in file:
                start_index = file.rfind("result_")
                end_index = file.rfind(".txt")
                
                model_name = file[start_index+7:end_index]
                if "logistic_reg" in model_name:
                    model_name_list.append("Logistic regression")
                if "randomForest" in model_name:
                    model_name_list.append("Random Forest")
                if "decision_tree" in model_name:
                    model_name_list.append("Decision Tree")
                if "naive_bayes" in model_name:
                    model_name_list.append("Naive Bayes")
                if "knn" in model_name:
                    model_name_list.append("k-NN")
                    
                print(model_name)
                with open(file) as in_file:
                    for line in in_file:
                        if len(line.strip())>0:
#                            print(line)
                            if "model" in line:
                                start_index = line.rfind("_")
                                disk_model = line[start_index+1:]
#                                print(disk_model)
                                if disk_model.strip() not in disk_model_dict:
                                    disk_model_dict[disk_model.strip()]=k
                                    rev_dict[k]=disk_model.strip()
                                    k+=1
                            if "Recall" in line:
                                start_index = line.rfind(":")
                                recall_value = float(line[start_index+1:])*100
                                index = disk_model_dict[disk_model.strip()]
                                print(index)
                                resultList[index].append(recall_value)
#        print(disk_model_dict)
        print(resultList)
        for x in range (len(resultList)):
            import matplotlib.pyplot as plt
            try:
                fig = plt.figure(figsize=(12, 6), dpi=100)
#                ax = fig.add_axes([0,0,1,1])
                width = 0.35
                plt.bar(model_name_list, resultList[x], width)
                
                plt.savefig("/home/masudulhasanmasudb/Music/hdd_data/model_comparison/"+str(day_number)+"/"+str(rev_dict[x])+".png")
                plt.show()
            except:
                print("error")
#            
            
#parent_folder_name = "/home/masudulhasanmasudb/Music/hdd_data/result_2017/"
#folder_list=glob.glob(parent_folder_name+"*")
##print(folder_list)
#file_list=[]
#for folder in folder_list:
#    if "/1" in folder:
#        start_index = folder.find("_2017")
#        day_number = folder[start_index+6:]
#        print(day_number)
#        file_list=glob.glob(folder+"/*")
#    #    print(file_list)
#        count=0
#        fileOpen=0
#        disk_model_list=[]
#        for file in file_list:
#            if ".txt" in file and "svm" not in file:
##                print(file)
#                with open(file) as in_file:
#                    fileOpen+=1
#                    for line in in_file:
#                        if len(line.strip())>0:
#                            if "model" in line:
#                                start_index = line.rfind(":")
#                                disk_model = line[start_index+1:].strip()
#                                if disk_model not in disk_model_list:
#                                    disk_model_list.append(disk_model)
##        print(disk_model_list)
##        print(file_list)
#        
#        for disk_model in disk_model_list:
#            
#            recall_value_list =[]
#            accuracy_value_list =[]
#            model_list=[]
#            for file in file_list:
#                if ".txt" in file and "svm" not in file and "knn" not in file:
#                    can_read= False
#                    start_index = file.rfind("result_")
#                    end_index = file.rfind(".txt")  
#                    model_name = file[start_index+7:end_index]
##                    print(model_name)
#                    with open(file) as in_file: 
#                        for line in in_file:
#                            if len(line.strip())>0:
#                                if disk_model in line:
#                                    can_read = True
##                                    print(file)
##                                    print(line)
#                                if "Recall" in line and can_read:
#                                    start_index = line.rfind(":")
#                                    recall_value = float(line[start_index+1:])*100
##                                    print(recall_value)
#                                    recall_value_list.append(recall_value)
##                                    model_list.append(model_name)
#                                    
#                                    if "logistic_reg" in model_name:
#                                        model_list.append("Logistic regression")
#                                    if "randomForest" in model_name:
#                                        model_list.append("Random Forest")
#                                    if "decision_tree" in model_name:
#                                        model_list.append("Decision Tree")
#                                    if "naive_bayes" in model_name:
#                                        model_list.append("Naive Bayes")
#                                    if "knn" in model_name:
#                                        model_list.append("k-NN")
#                                    
#                                    break
#                                
#                                if "Accuracy" in line and can_read:
#                                    start_index = line.rfind(":")
#                                    accuracy = float(line[start_index+1:])*100
#                                    accuracy_value_list.append(accuracy)
##                                    print(accuracy)
##                                    model_list.append(model_name)
#            print(disk_model)
#            print(recall_value_list)
#            print(accuracy_value_list)
#            print(model_list)
#            
#            recall_value_list = [float("%.2f"%item) for item in recall_value_list]
#            accuracy_value_list = [float("%.2f"%item) for item in accuracy_value_list]
#            
#            x = np.arange(len(model_list))  # the label locations
#            width = 0.35  # the width of the bars
#            
#            from pylab import rcParams
#            rcParams['figure.figsize'] = 16, 10
#            fig, ax = plt.subplots()
#            rects1 = ax.bar(x - width/2, accuracy_value_list, width, label='Accuracy')
#            rects2 = ax.bar(x + width/2, recall_value_list, width, label='Recall')
#            
#            # Add some text for labels, title and custom x-axis tick labels, etc.
#            ax.set_ylabel('Accuracy/Recall')
#            ax.set_title('Scores by group and gender')
#            ax.set_xticks(x)
#            ax.set_xticklabels(model_list)
#            ax.legend()
#            
#            
#            def autolabel(rects):
#                """Attach a text label above each bar in *rects*, displaying its height."""
#                for rect in rects:
#                    height = rect.get_height()
#                    ax.annotate('{}'.format(height),
#                                xy=(rect.get_x() + rect.get_width() / 2, height),
#                                xytext=(0, 3),  # 3 points vertical offset
#                                textcoords="offset points",
#                                ha='center', va='bottom')
#            
#            
#            autolabel(rects1)
#            autolabel(rects2)
#            
#            fig.tight_layout()
#            plt.savefig("/home/masudulhasanmasudb/Music/hdd_data/model_comparison/"+str(day_number)+"/"+str(disk_model)+"_comaprison.png")
#            plt.show()

                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                
        