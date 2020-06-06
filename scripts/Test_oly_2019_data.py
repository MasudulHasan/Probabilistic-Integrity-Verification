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

day=1
parent_folder_name = "../final_dataset/"+str(day)+"/"
folder_list=glob.glob(parent_folder_name+"*")
out_file = open("../result_2019/result_"+str(day)+"_only_2019_data.txt","a+")
# print(folder_list)
# selected_models = ['ST4000DM000', 'ST8000DM002', 'ST12000NM0007', 'ST8000NM0055', 'ST3000DM001', 'ST4000DX000']
selected_models = ['ST4000DM000', 'ST8000DM002', 'ST12000NM0007', 'ST8000NM0055']
for x in folder_list:
  try:
    start_index = x.rfind("/")
    end_index = x.rfind(".")
    model_name = x[start_index+1:end_index]
    if model_name in selected_models:
      hdd = pd.read_csv("../test_file_for_2019/1/"+model_name+".csv", header=None)
      hdd = hdd.drop(hdd.columns[6], axis=1)
      hdd = hdd.drop(hdd.columns[9], axis=1)
      hdd = hdd.drop(hdd.columns[14], axis=1)
      hdd = hdd.drop(hdd.columns[13], axis=1)
      hdd_pos = hdd.loc[hdd[hdd.columns[12]] == 1]
      hdd_neg = hdd[hdd[hdd.columns[12]] == 0]
      del hdd
      gc.collect()
      
      
      pos_train, pos_test = train_test_split(hdd_pos, test_size=0.2)
      neg_train, neg_test = train_test_split(hdd_neg, test_size=0.2)
      
      train_merged = [pos_train, neg_train]
      train_dataset = pd.concat(train_merged)

      train_dataset = train_dataset.dropna()

      x_new = train_dataset.iloc[:, :-1].values
      y_new = train_dataset.iloc[:, -1].values

      from imblearn.under_sampling import RandomUnderSampler
      rus = RandomUnderSampler()

      X_resampled, y_resampled = rus.fit_resample(x_new, y_new)
      print(Counter(y_resampled))
      
      clf= RandomForestClassifier(n_estimators=100)
      clf.fit(X_resampled,y_resampled)
      
      test_merged = [pos_test, neg_test]
      test_hdd1 = pd.concat(test_merged)
      
      test_hdd1 = test_hdd1.dropna()
      
      X_test = test_hdd1.iloc[:, :-1].values
      y_test = test_hdd1.iloc[:, -1].values
      y_pred=clf.predict(X_test)
      preds = clf.predict_proba(X_test)
      prob_score = calculate_accuracy(y_pred,y_test,preds)
      
      print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
      print("Recall: ",metrics.recall_score(y_test, y_pred))
      
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


      
  except:
      print(x)
      traceback.print_exc()