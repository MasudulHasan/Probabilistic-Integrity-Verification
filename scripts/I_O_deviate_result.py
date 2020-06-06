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

def get_error_score_list(pred_list):
    out_list=[]
    for x in pred_list:
        out_list.append(x[1])
    return out_list

selected_models = ['ST8000DM002', 'ST4000DX000', 'ST12000NM0007', 'ST4000DM000','ST8000NM0055','ST3000DM001']
for day in range(1,8):
  parent_folder_name = "../final_dataset/"+str(day)+"/"
  folder_list=glob.glob(parent_folder_name+"*")
  for x in folder_list:
    try:
      start_index = x.rfind("/")
      end_index = x.rfind(".")
      model_name = x[start_index+1:end_index]
      if model_name in selected_models:
        print(model_name)
        hdd = pd.read_csv(x, header=None)
        # print(hdd.describe)
        train_dataset, test_dataset = train_test_split(hdd, test_size=0.2)

        del hdd
        gc.collect()

        train_bit_error_label = train_dataset[14]
        train_sector_error_label = train_dataset[15]
        train_disk_failure_label = train_dataset[16]

        test_bit_error_label = test_dataset[14]
        test_sector_error_label = test_dataset[15]
        test_disk_failure_label = test_dataset[16]

        train_dataset = train_dataset.drop(train_dataset.columns[16], axis=1)
        train_dataset = train_dataset.drop(train_dataset.columns[14], axis=1)
        train_dataset = train_dataset.drop(train_dataset.columns[14], axis=1)
        train_dataset = train_dataset.drop(train_dataset.columns[10], axis=1)

        test_dataset = test_dataset.drop(test_dataset.columns[16], axis=1)
        test_dataset = test_dataset.drop(test_dataset.columns[14], axis=1)
        test_dataset = test_dataset.drop(test_dataset.columns[14], axis=1)
        test_dataset = test_dataset.drop(test_dataset.columns[10], axis=1)

        columns = dict(map(reversed, enumerate(train_dataset.columns)))
        train_dataset = train_dataset.rename(columns=columns)
        test_dataset = test_dataset.rename(columns=columns)

        # Disk Failure 
        train_dataset[13] = train_disk_failure_label
        train_dataset = train_dataset.dropna()

        x_new = train_dataset.iloc[:, :-1].values
        y_new = train_dataset.iloc[:, -1].values

        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler()

        X_resampled, y_resampled = rus.fit_resample(x_new, y_new)
        print(Counter(y_resampled))

        clf= RandomForestClassifier(n_estimators=100)
        clf.fit(X_resampled,y_resampled)

        test_dataset[13] = test_disk_failure_label
        test_dataset = test_dataset.dropna()
                    
        X_test = test_dataset.iloc[:, :-1].values
        y_test = test_dataset.iloc[:, -1].values
        y_pred=clf.predict(X_test)
        preds = clf.predict_proba(X_test)
                    
        prob_score = calculate_accuracy(y_pred,y_test,preds)
        failure_error_list = get_error_score_list(preds)
        
        # print(x)                
        print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
        print("Recall: ",metrics.recall_score(y_test, y_pred))

        out_file = open("../I_O_deviate_result/"+str(model_name)+"_"+str(day)+"_failure.txt","a+")
        out_file.write("\n\n"+"model: "+model_name+"\n")
        out_file.write("\n")
        out_file.write("Accuracy: "+ str(metrics.accuracy_score(y_test, y_pred))+"\n")
        out_file.write("Recall: "+ str(metrics.recall_score(y_test, y_pred))+"\n")
        conf_matrix = metrics.confusion_matrix(y_test, y_pred)
        out_file.write(str(conf_matrix))
        out_file.write("\n")
        out_file.write(str(perf_measure(y_test, y_pred)))
        out_file.write("\n")
        out_file.write(prob_score)
        out_file.flush()
        out_file.close()

        del X_resampled
        del y_resampled
        del X_test
        del y_test
        del x_new
        del y_new
        del preds
        del prob_score
        gc.collect()

        # Sector_error
        train_dataset = train_dataset.drop(train_dataset.columns[13], axis=1)
        train_dataset[13] = train_sector_error_label
        train_dataset = train_dataset.dropna()

        x_new = train_dataset.iloc[:, :-1].values
        y_new = train_dataset.iloc[:, -1].values

        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler()

        X_resampled, y_resampled = rus.fit_resample(x_new, y_new)
        print(Counter(y_resampled))

        clf= RandomForestClassifier(n_estimators=100)
        clf.fit(X_resampled,y_resampled)

        test_dataset = test_dataset.drop(test_dataset.columns[13], axis=1)
        test_dataset[13] = test_sector_error_label
        test_dataset = test_dataset.dropna()
                    
        X_test = test_dataset.iloc[:, :-1].values
        y_test = test_dataset.iloc[:, -1].values
        y_pred=clf.predict(X_test)
        preds = clf.predict_proba(X_test)
                    
        prob_score = calculate_accuracy(y_pred,y_test,preds)
        sector_error_list = get_error_score_list(preds)
        
        # print(x)                
        print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
        print("Recall: ",metrics.recall_score(y_test, y_pred))

        out_file = open("../I_O_deviate_result/"+str(model_name)+"_"+str(day)+"_sector.txt","a+")
        out_file.write("\n\n"+"model: "+model_name+"\n")
        out_file.write("\n")
        out_file.write("Accuracy: "+ str(metrics.accuracy_score(y_test, y_pred))+"\n")
        out_file.write("Recall: "+ str(metrics.recall_score(y_test, y_pred))+"\n")
        conf_matrix = metrics.confusion_matrix(y_test, y_pred)
        out_file.write(str(conf_matrix))
        out_file.write("\n")
        out_file.write(str(perf_measure(y_test, y_pred)))
        out_file.write("\n")
        out_file.write(prob_score)
        out_file.flush()
        out_file.close()

        del X_resampled
        del y_resampled
        del X_test
        del y_test
        del x_new
        del y_new
        del preds
        del prob_score
        gc.collect()

        # print(sector_error_list)

        #Bit error 

        # Sector_error
        train_dataset = train_dataset.drop(train_dataset.columns[13], axis=1)
        train_dataset[13] = train_bit_error_label
        train_dataset = train_dataset.drop(train_dataset.columns[6], axis=1)
        train_dataset = train_dataset.dropna()

        x_new = train_dataset.iloc[:, :-1].values
        y_new = train_dataset.iloc[:, -1].values

        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler()

        X_resampled, y_resampled = rus.fit_resample(x_new, y_new)
        print(Counter(y_resampled))

        clf= RandomForestClassifier(n_estimators=100)
        clf.fit(X_resampled,y_resampled)

        test_dataset = test_dataset.drop(test_dataset.columns[13], axis=1)
        test_dataset[13] = test_bit_error_label
        test_dataset = test_dataset.drop(test_dataset.columns[6], axis=1)
        test_dataset = test_dataset.dropna()
                    
        X_test = test_dataset.iloc[:, :-1].values
        y_test = test_dataset.iloc[:, -1].values
        y_pred=clf.predict(X_test)
        preds = clf.predict_proba(X_test)
                    
        prob_score = calculate_accuracy(y_pred,y_test,preds)
        bit_error_list = get_error_score_list(preds)
        
        # print(x)                
        print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
        print("Recall: ",metrics.recall_score(y_test, y_pred))

        out_file = open("../I_O_deviate_result/"+str(model_name)+"_"+str(day)+"_bit.txt","a+")
        out_file.write("\n\n"+"model: "+model_name+"\n")
        out_file.write("\n")
        out_file.write("Accuracy: "+ str(metrics.accuracy_score(y_test, y_pred))+"\n")
        out_file.write("Recall: "+ str(metrics.recall_score(y_test, y_pred))+"\n")
        conf_matrix = metrics.confusion_matrix(y_test, y_pred)
        out_file.write(str(conf_matrix))
        out_file.write("\n")
        out_file.write(str(perf_measure(y_test, y_pred)))
        out_file.write("\n")
        out_file.write(prob_score)
        out_file.flush()
        out_file.close()

        del X_resampled
        del y_resampled
        del X_test
        del y_test
        del x_new
        del y_new
        del preds
        del prob_score
        gc.collect()

        # print(bit_error_list)

        out_file = open("../I_O_deviate_result/"+str(model_name)+"_"+str(day)+".csv","a+")
        for index in range(len(bit_error_list)):
          out_file.write(str(bit_error_list[index])+","+str(sector_error_list[index])+","+str(failure_error_list[index]))
          out_file.write("\n")
          out_file.flush()
        out_file.close()
        
        del bit_error_list
        del sector_error_list
        del failure_error_list

        gc.collect()

    except:
      # print(x)
      traceback.print_exc()