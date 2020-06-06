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

map_list = []
index_map = {}
date_dict = {}
now = time.time()
count = -1
file_name = ""
with open("../map_2019.txt", "r")as in_file:
    for line in in_file:
        if(len(line.strip()) > 0):
            if".csv" in line:
                if(count != -1):
                    map_list.append(date_dict)
                    index_map[file_name] = count

#                date_dict.clear()
                date_dict = {}
                count += 1
                file_name = line.strip()
            else:
                parts = line.strip().split(" ")
                date_dict[parts[0]] = int(float(parts[1]))

print(count)


def get_lable(serial_number_list, date, year, month, day):
    global parent_folder_name
    next_day_label = {}

    now = datetime.datetime(year, month, day)
    next_day = (now + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    for x in serial_number_list:
        try:
            index = index_map[x+".csv"]
            current_value = map_list[index][date]
            next_day_value = map_list[index][next_day]


            if(current_value == next_day_value):
                next_day_label[x] = 0
            else:
                next_day_label[x]= 1
        except:
            traceback.print_exc()

    return next_day_label


number_of_days = 1
disk_model_name = sys.argv[1]

features = [1, 4, 5, 7, 9, 12, 188, 193, 194, 197, 198, 199]
columns_specified = []
for feature in features:
    columns_specified += ["smart_{0}_raw".format(feature)]
columns_specified = ["serial_number"] + columns_specified
hdd = pd.read_csv("../final_dataset/"+str(number_of_days) +
                  '/'+str(disk_model_name)+'.csv', header=None)
#hdd = pd.read_csv("../dataset_1.csv")
hdd = hdd.drop(hdd.columns[6], axis=1)
hdd = hdd.drop(hdd.columns[9], axis=1)
hdd = hdd.drop(hdd.columns[14], axis=1)
hdd = hdd.drop(hdd.columns[13], axis=1)
hdd = hdd.dropna()
#    print(hdd.head())

month_to_be_considerate = 1

while(month_to_be_considerate <= 9):
    prev_month = 1
    df_list = []
    year_to_cosider=2019
    while(prev_month < month_to_be_considerate):
        hdd_extra = pd.read_csv(
            "../month_wise_file/"+str(year_to_cosider)+"/"+str(prev_month)+'/'+str(disk_model_name)+'.csv', header=None)
        hdd_extra = hdd_extra.drop(hdd_extra.columns[6], axis=1)
        hdd_extra = hdd_extra.drop(hdd_extra.columns[9], axis=1)
        # hdd_extra = hdd_extra.drop(hdd_extra.columns[14], axis=1)
        # hdd_extra = hdd_extra.drop(hdd_extra.columns[13], axis=1)
        hdd_extra = hdd_extra.dropna()
        df_list.append(hdd_extra)
        prev_month += 1

    df_list.append(hdd)
    result = pd.concat(df_list)

    x = result.iloc[:, :-1].values
    y = result.iloc[:, -1].values

    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(.5)

    X_resampled, y_resampled = rus.fit_resample(x, y)
    print(Counter(y_resampled))
    clf = RandomForestClassifier()
    clf.fit(X_resampled, y_resampled)

    for day in range(1, 32):
        try:
            month_str = "0"+str(month_to_be_considerate)
            if day <= 9:
                day_str = "0"+str(day)
            else:
                day_str = str(day)
    
            correctDate = None
            try:
                newDate = datetime.datetime(2019, month_to_be_considerate, day)
                correctDate = True
            except ValueError:
                correctDate = False
            if correctDate == True:
                date = str(2019)+"-"+month_str+"-"+day_str
                print(date)
                df = pd.read_csv("../data/"+date+".csv")
                df = df.loc[df['model'] == str(disk_model_name)]
                df = df[columns_specified]
                df = df.dropna()
                serial_number = df.iloc[:, 0].values
                df = df.drop(df.columns[0], axis=1)
                print(df.columns)
                pred_value = clf.predict(df)
                preds = clf.predict_proba(df)
                next_day_label = get_lable(serial_number, date, 2019, month_to_be_considerate, day)
    
                print(len(pred_value))
                print(len(preds))
                print(len(next_day_label))
    
                output_file = open("../predicted_result_"+disk_model_name+"/"+str(date)+".csv", "a+")
                for index in range(len(serial_number)):
                    if serial_number[index] in next_day_label:
                        output_file.write(str(pred_value[index])+","+str(preds[:, 0][index])+","+str(preds[:, 1][index])+","+str(next_day_label[serial_number[index]])+"\n")
                        output_file.flush()
                output_file.close()
        except:
            traceback.print_exc()

    month_to_be_considerate += 1
