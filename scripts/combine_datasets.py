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
import sys, traceback
import threading
import datetime
from random import *
from collections import defaultdict
import collections

# year=2015
# while(year<=2018):
#     df = pd.read_csv("dataset_"+str(year)+"/1/ST4000DM000.csv",header=None)
#     output_file = open("final_dataset_corrected/"+"ST4000DM000.csv","a+")
#     df.to_csv(output_file, header=False, index=False)
    
#     year+=1

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    
    # print(TP/(TP+FN)) 
    
    # print(FP/(FP+TN))       
           
    return(TP, FP, TN, FN)
    
def calculate_accuracy(y_pred, y_test, pred_list):
    output_str=""
    for x in range(len(y_pred)):
        output_str+=str(y_pred[x])+" "+str(y_test[x])+" "+str(pred_list[x])+"\n"
    
    return output_str


disk_model_name = "ST4000DM000"

hdd = pd.read_csv("../final_dataset_corrected/"+str(disk_model_name)+'.csv', header=None)
hdd = hdd.drop(hdd.columns[6], axis=1)
hdd = hdd.drop(hdd.columns[9], axis=1)
hdd = hdd.drop(hdd.columns[14], axis=1)
hdd = hdd.drop(hdd.columns[13], axis=1)
hdd = hdd.dropna()

x = hdd.iloc[:, :-1].values
y = hdd.iloc[:, -1].values

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler()

X_resampled, y_resampled = rus.fit_resample(x, y)
print(Counter(y_resampled))
clf = RandomForestClassifier()
clf.fit(X_resampled, y_resampled)

year_to_cosider=2019
hdd_extra = pd.read_csv("../month_wise_file/"+str(year_to_cosider)+"/"+str(1)+'/'+str(disk_model_name)+'.csv', header=None)
hdd_extra = hdd_extra.drop(hdd_extra.columns[6], axis=1)
hdd_extra = hdd_extra.drop(hdd_extra.columns[9], axis=1)
# hdd_extra = hdd_extra.drop(hdd_extra.columns[14], axis=1)
# hdd_extra = hdd_extra.drop(hdd_extra.columns[13], axis=1)
hdd_extra = hdd_extra.dropna()

x_test = hdd_extra.iloc[:, :-1].values
y_test = hdd_extra.iloc[:, -1].values

y_pred = clf.predict(x_test)
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
print("Recall: ",metrics.recall_score(y_test, y_pred))
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(conf_matrix)

print(perf_measure(y_test, y_pred))