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
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# map_list = []
# index_map = {}
# date_dict = {}
# now = time.time()
# count = -1
# file_name = ""
# with open("../map_2019.txt", "r")as in_file:
#     for line in in_file:
#         if(len(line.strip()) > 0):
#             if".csv" in line:
#                 if(count != -1):
#                     map_list.append(date_dict)
#                     index_map[file_name] = count

# #                date_dict.clear()
#                 date_dict = {}
#                 count += 1
#                 file_name = line.strip()
#             else:
#                 parts = line.strip().split(" ")
#                 date_dict[parts[0]] = int(float(parts[1]))

# print(count)


# def get_lable(serial_number_list, date, year, month, day):
#     next_day_label = {}

#     now = datetime.datetime(year, month, day)
#     next_day = (now + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

#     for x in serial_number_list:
#         try:
#             index = index_map[x+".csv"]
#             current_value = map_list[index][date]
#             next_day_value = map_list[index][next_day]


#             if(current_value == next_day_value):
#                 next_day_label[x] = 0
#             else:
#                 next_day_label[x]= 1
#         except:
#             # traceback.print_exc()
#             pass

#     return next_day_label

features = [1, 4, 5, 7, 9, 12, 187, 188, 193, 194, 197, 198, 199]
columns_specified = []
for feature in features:
    columns_specified += ["smart_{0}_raw".format(feature)]
columns_specified = ["serial_number"] + columns_specified

# selected_models = ['ST12000NM0007','ST4000DM000','ST8000DM002','ST8000NM0055']
selected_models = ['ST6000DX000', 'ST10000NM0086']
# for iter_ in range(5):
model = 'ST12000NM0007'
hdd1 = pd.read_csv("../final_dataset/"+str(1)+'/'+str(model)+'.csv', header=None)
hdd1 = hdd1.drop(hdd1.columns[10], axis=1)
hdd1 = hdd1.drop(hdd1.columns[15], axis=1)
hdd1 = hdd1.drop(hdd1.columns[14], axis=1)
hdd1 = hdd1.dropna()
# print(hdd.head())
month_to_be_considerate = 11
t_recall=0.0
t_count=0
while(month_to_be_considerate <= 12):

    t_recall_month=0.0
    t_count_month=0

    prev_month = 1
    year_to_cosider=2018
    # prev_month = month_to_be_considerate-6
    # if(month_to_be_considerate<7):
    #     year_to_cosider=2018
    #     prev_month+=12
    df_list = []
    
    # while(prev_month < month_to_be_considerate):
    # # for tm in range(6):
    #     hdd_extra = pd.read_csv("../month_wise_file/"+str(year_to_cosider)+"/"+str(prev_month)+'/'+str(model)+'.csv', header=None)
    #     hdd_extra = hdd_extra.drop(hdd_extra.columns[10], axis=1)
    #     #hdd_extra = hdd_extra.drop(hdd_extra.columns[9], axis=1)
    #     hdd_extra = hdd_extra.dropna()
    #     df_list.append(hdd_extra)
    #     prev_month += 1

    #     if(prev_month>12):
    #         prev_month=1
    #         year_to_cosider+=1

    df_list.append(hdd1)
    result = pd.concat(df_list)

    x = result.iloc[:, :-1].values
    y = result.iloc[:, -1].values
    print(len(y))

    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler()

    X_resampled, y_resampled = rus.fit_resample(x, y)
    print(Counter(y_resampled))
    clf = RandomForestClassifier()
    clf.fit(X_resampled, y_resampled)

    # model = "NN"
    # clf = make_pipeline(StandardScaler(), MLPClassifier())
    # clf.fit(X_resampled,y_resampled)

    for test_model in selected_models:
        out_file = out_file = open("../transfer_learning_result/"+test_model+".txt","a+")
        hdd_test = pd.read_csv("../month_wise_file/"+str(year_to_cosider)+"/"+str(month_to_be_considerate)+'/'+str(test_model)+'.csv', header=None)
        hdd_test = hdd_test.drop(hdd_test.columns[10], axis=1)
        hdd_test = hdd_test.dropna()
    
        X_test = hdd_test.iloc[:, :-1].values
        y_test = hdd_test.iloc[:, -1].values

        y_pred=clf.predict(X_test)
        preds = clf.predict_proba(X_test)

        for index in range(len(y_test)):
            out_file.write(str(y_pred[index])+","+str(preds[:, 0][index])+","+str(preds[:, 1][index])+","+str(y_test[index])+"\n")
            out_file.flush()
        
        # out_file.write("\n")
        # out_file.flush()

        del hdd_test
        gc.collect()

        out_file.close()

    del result
    gc.collect()
        # for test_model in selected_models:
        #     for day in range(1, 32):
        #         try:
        #             month_str = "0"+str(month_to_be_considerate)
        #             if day <= 9:
        #                 day_str = "0"+str(day)
        #             else:
        #                 day_str = str(day)
            
        #             correctDate = None
        #             try:
        #                 newDate = datetime.datetime(2019, month_to_be_considerate, day)
        #                 correctDate = True
        #             except ValueError:
        #                 correctDate = False
        #             if correctDate == True:
        #                 date = str(2019)+"-"+month_str+"-"+day_str
        #                 # print(date)
        #                 df = pd.read_csv("../data/"+date+".csv")
        #                 df = df.loc[df['model'] == str(test_model)]
        #                 df = df[columns_specified]
        #                 df = df.dropna()
        #                 serial_number = df.iloc[:, 0].values
        #                 df = df.drop(df.columns[0], axis=1)
        #                 # print(df.columns)
        #                 pred_value = clf.predict(df)
        #                 preds = clf.predict_proba(df)
        #                 next_day_label = get_lable(serial_number, date, 2019, month_to_be_considerate, day)
            
        #                 # print(len(pred_value))
        #                 # print(pred_value)
        #                 # print(len(next_day_label))
        #                 # print(pred_value[0])
        #                 tp=0
        #                 tn=0
        #                 fp=0
        #                 fn=0 
                        
        #                 pred_value_new=[]
        #                 next_day_value_new=[]

        #                 for index in range(len(serial_number)):
        #                     if serial_number[index] in next_day_label:
        #                         if pred_value[index]==0 and next_day_label[serial_number[index]]==0:
        #                             tn+=1
        #                         elif pred_value[index]==1 and next_day_label[serial_number[index]]==1:
        #                             tp+=1
        #                         elif pred_value[index]==1 and next_day_label[serial_number[index]]==0:
        #                             fp+=1
        #                         elif pred_value[index]==0 and next_day_label[serial_number[index]]==1:
        #                             fn+=1

        #                 try:
        #                     # print("recall ", (tp/(tp+fn))*100)
        #                     t_recall+=((tp/(tp+fn))*100)
        #                     t_recall_month+=((tp/(tp+fn))*100)
        #                     t_count+=1
        #                     t_count_month+=1
        #                 except:
        #                     pass
        #                     # print("No error")
        #         except:
        #             traceback.print_exc()



        # print("MOnth ",month_to_be_considerate, t_recall_month/t_count_month)
    month_to_be_considerate+=1
    # print(t_recall/t_count)

