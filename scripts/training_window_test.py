import time
import glob
import random
import datetime
import os
import subprocess
import shlex
import gc
import pandas as pd
import collections
import numpy as np
import pandas as pd
import time
# import ray
import psutil
import multiprocessing
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import ensemble, metrics
import gc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from imblearn.over_sampling import (RandomOverSampler, SMOTE, ADASYN)
from collections import Counter
import sys
import traceback
import threading
import datetime
from random import *

selected_models = ['ST4000DM000', 'ST8000DM002', 'ST12000NM0007', 'ST8000NM0055']
features = [1, 4, 5, 7, 9, 12, 188, 193, 194, 197, 198, 199]
columns_specified = []
for feature in features:
    columns_specified += ["smart_{0}_raw".format(feature)]
columns_specified = ["serial_number"] + columns_specified


#    print(hdd.head())

for disk_model_name in selected_models:
    hdd = pd.read_csv("../final_dataset/"+str(1) + '/'+str(disk_model_name)+'.csv', header=None)
    #hdd = pd.read_csv("../dataset_1.csv")
    hdd = hdd.drop(hdd.columns[6], axis=1)
    hdd = hdd.drop(hdd.columns[9], axis=1)
    hdd = hdd.drop(hdd.columns[14], axis=1)
    hdd = hdd.drop(hdd.columns[13], axis=1)
    hdd = hdd.dropna()
    month_to_be_considerate = 7
    while(month_to_be_considerate <= 9):
        lst=[1,3,4,6,9]
        for window in lst:
            df_list = []
            considered_so_far=0
            year_to_cosider = 2019
            prev_month = month_to_be_considerate-1
            while(considered_so_far < window):
                if prev_month==0:
                    break               
                hdd_extra = pd.read_csv("../month_wise_file/"+str(year_to_cosider)+"/"+str(prev_month)+'/'+str(disk_model_name)+'.csv', header=None)
                hdd_extra = hdd_extra.drop(hdd_extra.columns[6], axis=1)
                hdd_extra = hdd_extra.drop(hdd_extra.columns[9], axis=1)
                hdd_extra = hdd_extra.dropna()
                df_list.append(hdd_extra)
                prev_month -= 1
                considered_so_far+=1

            if window==9:
                df_list.append(hdd)
            
            result = pd.concat(df_list)

            x = result.iloc[:, :-1].values
            y = result.iloc[:, -1].values

            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler()

            X_resampled, y_resampled = rus.fit_resample(x, y)
            print(Counter(y_resampled))
            clf = RandomForestClassifier()
            clf.fit(X_resampled, y_resampled)

            hdd_test = pd.read_csv("../month_wise_file/"+str(year_to_cosider)+"/"+str(month_to_be_considerate)+'/'+str(disk_model_name)+'.csv', header=None)
            hdd_test = hdd_test.drop(hdd_test.columns[6], axis=1)
            hdd_test = hdd_test.drop(hdd_test.columns[9], axis=1)
            hdd_test = hdd_test.dropna()

            X_test = hdd_test.iloc[:, :-1].values
            y_test = hdd_test.iloc[:, -1].values
            y_pred=clf.predict(X_test)
            preds = clf.predict_proba(X_test)
            print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
            print("Recall: ",metrics.recall_score(y_test, y_pred))
            conf_matrix = metrics.confusion_matrix(y_test, y_pred)
            print(conf_matrix)
            print(metrics.roc_auc_score(y_test, y_pred))
            # print(perf_measure(y_test, y_pred))
            out_file = open("../training_window_test_result/"+disk_model_name+"_"+str(month_to_be_considerate)+"_"+str(window)+".txt","a+")
            for index in range(len(y_test)):
                out_file.write(str(y_test[index])+"\t"+str(y_pred[index])+"\t"+str(preds[index])+"\n")
                out_file.flush()
            out_file.close()
        
        month_to_be_considerate+=1