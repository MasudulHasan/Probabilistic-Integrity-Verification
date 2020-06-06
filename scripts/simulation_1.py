#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 19:35:47 2019

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

start_year = sys.argv[0]
#end_year = sys.argv[1]

parent_folder_name = "output_files/"
folder_list=glob.glob(parent_folder_name+"*")

map_list = []
index_map ={}
date_dict={}
now = time.time()
count=-1
file_name=""
with open("map1.txt","r")as in_file:
        for line in in_file:
            if(len(line.strip())>0):
                if".csv" in line:
                    if(count!=-1):
                        map_list.append(date_dict)
                        index_map[file_name]=count
                    
                    date_dict.clear()
                    count+=1
                    file_name=line.strip()
                else:
                    parts = line.strip().split(" ")
                    date_dict[parts[0]] = int(parts[1])
                    
#                print(line)
                
                
print(count)
#print(map_list[1])
#class fileReadThread(threading.Thread):
#    def __init__(self, start_index,folder_list):
#        threading.Thread.__init__(self)
#        self.start_index = start_index
#        self.folder_list = folder_list
#        self.map_list = []
#        self.index_map ={}
#
#    def run(self):
#        count = 0
#        end_index = min(self.start_index+500, len(folder_list))
#        print("end index "+str(end_index))
#        for x in range(self.start_index,end_index):
#            df = pd.read_csv(self.folder_list[x])
#            dates = df['date'].to_list()
#    #        print(dates)
#            date_dict={}
#            for date in dates:
#                value = df.loc[df['date'] == date, 'smart_187_raw'].iloc[0]
#                date_dict[date] = value
#                
#            self.map_list.append(date_dict)
#            self.index_map[x]=count
#       
#        print("start index "+str(self.start_index))
##        print(self.map_list)
#        print(self.index_map)
#
#def make_lable():
#    global folder_list
#    count =0
#    for x in folder_list:
#        df = pd.read_csv(x)
#        dates = df['date'].to_list()
##        print(dates)
#        date_dict={}
#        for date in dates:
#            value = df.loc[df['date'] == date, 'smart_187_raw'].iloc[0]
#            date_dict[date] = value
#            
#        map_list.append(date_dict)
#        index_map[x]=count
##        count+=1
##        if(count==100):
##            break
##        current_value = df.loc[df['date'] == date, 'smart_187_raw'].iloc[0]
    

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
             
    return(TP, FP, TN, FN)

def count_unique(keys):
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return uniq_keys, np.bincount(bins)


##out_file = open("done_so_far.txt","a+")
#print(len(folder_list))


def get_lable(serial_number_list,date,year,month,day):
    global parent_folder_name
    next_day_label =[]
    
    now = datetime.datetime(year,month,day)
    next_day = (now + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    for x in serial_number_list:
        index = index_map[x+".csv"]
        current_value = map_list[index][date]
        
        next_day_value = map_list[index][next_day]
        
        if(current_value==next_day_value):
            next_day_label.append(0)
        else:
            next_day_label.append(1)
        
    return next_day_label
        

def Sort_Tuple(tup):   
    return(sorted(tup, key = lambda x: x[0], reverse = True))


def calculate_accuracy(tuple_list, real_list):
    total = 0
    tf =0
    for x in range(len(tuple_list)):
        item = tuple_list[x]
        prob = float(item[0])
        if(prob<0.4):
            total+=1
            index = int(item[0])
            if real_list[index]==0:
                tf+=1
    
#    print(" calcuated accuracy = "+str((tf/total))) 

    return str((tf/total)), str(len(tuple_list)-total)           

#make_lable()
#print(index_map)
#print(time.time()-now) 

#x =0
#t=[]
#while x<=3000:
#    thread1 = fileReadThread(x,folder_list)
#    thread1.start()
#    t.append(thread1)
#    x+=500
#
#for x in t:
#    x.join()
    
    
#print(time.time()-now) 
           
hdd = pd.read_csv('dataset_1.csv')
hdd = hdd.drop(hdd.columns[6], axis=1)
hdd = hdd.drop(hdd.columns[9], axis=1)
hdd = hdd.dropna()
hdd_pos = hdd.loc[hdd[hdd.columns[12]] == 1]
hdd = hdd[hdd[hdd.columns[12]] == 0]
hdd_neg = hdd
hdd_merged = [hdd_pos, hdd_neg]
result = pd.concat(hdd_merged)
x = result.iloc[:, :-1].values
y = result.iloc[:, -1].values
X_resampled, y_resampled = SMOTE().fit_resample(x, y)
#X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)
clf=RandomForestClassifier()
clf.fit(X_resampled,y_resampled)


features = [1, 4, 5, 7, 9, 12, 188, 193, 194, 197, 198, 199]
columns_specified = []
for feature in features:
    	columns_specified += ["smart_{0}_raw".format(feature)]



error_file_list=[]
with open("files_with_error.txt","r")as in_file:
        for line in in_file:
            if(len(line.strip())>0):
                error_file_list.append(line.strip())

good_file_list = []
for x in folder_list:
    back_slash_index = x.rfind("/")
    file_name = x[back_slash_index+1:]
#    print(file_name)
    if file_name not in error_file_list:
        good_file_list.append(x)

#print(len(folder_list) - len(good_file_list)) 

test_error_file_list=[]
with open("files_for_test.txt","r")as in_file:
        for line in in_file:
            if(len(line.strip())>0):
                test_error_file_list.append(line.strip())

#print(len(test_error_file_list))

all_file_list = good_file_list+test_error_file_list
#print(len(all_file_list))
stripe_size = 50

output_file = open("final_log_19_11_19_"+str(start_year)+".txt","a+")
#
year = int(start_year);
end_year = year 
while year<=end_year:
    for month in range(1,13):
        for day in range(1,32):
            if month<=9:
                    month_str = "0"+str(month)
            else:
                month_str = str(month)
            
            if day<=9:
                day_str = "0"+str(day)
            else:
                day_str = str(day)
            
            import datetime
            correctDate = None
            try:
                newDate = datetime.datetime(year,month,day)
                correctDate = True
            except ValueError:
                correctDate = False
#            print(str(correctDate))
            
            if correctDate==True:
                try:
                    date = str(year)+"-"+month_str+"-"+day_str
                    print(date)
                    df = pd.read_csv("data/"+date+".csv")
#                    df = df.loc[df['model'] == 'ST4000DM000']
                    df = df.loc[df['serial_number'] !="S300XQ5W"]
                    
                    total_disk_number = 0
                    total_check_disk = 0

                    output_str =""
                    for numof_iter in range(4000):
                        try:
                    
                            file_size =int(random.getrandbits(7))
            #                print(file_size)
                            
                            
            #                print(df['model'])
            #                print(len(df))
                            
                            disk_number = int((file_size*1024)/500)
                            total_disk_number+=disk_number
            #                print(disk_number)
                            
                            selected_disk = df.sample(disk_number)
                            serial_number = selected_disk.iloc[:, 1].values
            #                print(serial_number)
                            selected_disk = selected_disk[columns_specified]
                            pred_value = clf.predict(selected_disk)
                            preds = clf.predict_proba(selected_disk)
                            
            #                print(pred_value)
        #                    print(preds[:,1])
        #                    print(preds[:,0])
                            
                            predicted_pair_list = []
                            
                            for x in range(len(preds[:,1])):
                                predicted_pair_list.append((preds[:,1][x],x))
                            
        #                    print(np.prod(preds[:,0]) )
        #                    print(1.0 - np.prod(preds[:,0]) )
                            
                            s_list = Sort_Tuple(predicted_pair_list)
                            
                            output_str+="\n\n"+str(date)+"\n"
                            output_str+="predicted_value: \n"
                            output_str+=str(pred_value) + "\n"
                            # output_file.write("probability: \n")
                            # output_file.write(str(preds) + "\n")


                            # output_file.write("\n\n"+str(date)+"\n")
                            #
                            # output_file.write("predicted_value: \n")
                            # output_file.write(str(pred_value)+"\n")
                            # output_file.write("probability: \n")
                            # output_file.write(str(preds)+"\n")
                            
                            
            #                print(selected_disk.head)
                            
                            next_day_label = get_lable(serial_number,date,year,month,day)

                            output_str+="next day real value: \n"
                            output_str+=str(next_day_label)+"\n"

                            # output_file.write("next day real value: \n")
                            # output_file.write(str(next_day_label)+"\n")
        #                    output_file.write("seven days value: \n")
        #                    output_file.write(str(seven_days_label)+"\n")
                            
                            
                            c_auuracy, checksum_disk = calculate_accuracy(s_list, next_day_label)
                            total_check_disk+=int(checksum_disk)

                            output_str += "self calculated accuracy "+ c_auuracy+"\n"
                            output_str += "cheksem run on "+ checksum_disk +"\n"

                            # output_file.write("self calculated accuracy "+ c_auuracy+"\n")
                            # output_file.write("cheksem run on "+ checksum_disk +"\n")
                            
                            
                            
                            # output_file.write("next day stat: \n")
        #                    print("N Accuracy: ",metrics.accuracy_score(next_day_label, pred_value))
        #                    print("N Recall: ",metrics.recall_score(next_day_label, pred_value))
        #                     output_file.write(str(metrics.accuracy_score(next_day_label, pred_value))+"\n")
        #                     output_file.write(str(metrics.recall_score(next_day_label, pred_value))+"\n")
        #                     output_file.flush()
            #                conf_matrix = metrics.confusion_matrix(y_test, y_pred)
            #                print(conf_matrix)
                            TP, FP, TN, FN = perf_measure(next_day_label, pred_value)
                            # output_file.write(str(TP)+" "+str(FP)+" "+str(TN)+" "+str(FN)+"\n")



                            output_str+="next day stat: \n"
                            output_str+= str(metrics.accuracy_score(next_day_label, pred_value))+"\n"
                            output_str+=str(metrics.recall_score(next_day_label, pred_value))+"\n"
                            output_str+="TP: "+str(TP)+" "+str(FP)+" "+str(TN)+" "+str(FN)+"\n"

                        except:
                            traceback.print_exc()

                    output_file.write(output_str)
                    output_file.write("total_dik "+ str(total_disk_number) +"\n")
                    output_file.write("check Sum run on "+ str(total_check_disk) +"\n")
                    output_file.write("perct "+ str(total_check_disk/total_disk_number) +"\n")
                    output_file.flush()
                    
                except:
                    traceback.print_exc()

    year+=1                   