#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 03:40:19 2020

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
    
    print(TP/(TP+FN)) 
    
    print(FP/(FP+TN))       
           
    return(TP, FP, TN, FN)
    
def calculate_accuracy(y_pred, y_test, pred_list):
    output_str=""
    for x in range(len(y_pred)):
#        if y_pred[x]!=y_test[x]:
        output_str+=str(y_pred[x])+" "+str(y_test[x])+" "+str(pred_list[x])+"\n"
    
    return output_str
#    print(" calcuated accuracy = "+str((tf/total))) 

#    return str((tf/total)), str(len(tuple_list)-total)     

count = 0
for day in range(1,4):
    parent_folder_name = "../final_dataset/"+str(day)+"/"
    folder_list=glob.glob(parent_folder_name+"*")
    out_file = open("../final_result/"+str(day)+"/RF_new_1.txt","a+")
    for x in folder_list:
        start_index = x.rfind("/")
        end_index = x.rfind(".")
        model_name = x[start_index+1:end_index]
        print(model_name)
#        if day==1 and "ST4000DM000" not in model_name:
        try:
            hdd = pd.read_csv(x, header=None)
        #    print(hdd.head)
            hdd = hdd.drop(hdd.columns[6], axis=1)
            hdd = hdd.drop(hdd.columns[9], axis=1)
            hdd = hdd.drop(hdd.columns[14], axis=1)
            hdd = hdd.drop(hdd.columns[13], axis=1)
            #print(hdd.describe())
            
#            hdd = hdd[:1000000]
            
            hdd_pos = hdd.loc[hdd[hdd.columns[12]] == 1]
            #print(len(hdd_pos))
            
            hdd_neg = hdd[hdd[hdd.columns[12]] == 0]
            #print(len(hdd_neg))
            
            #print(len(hdd))
            del hdd
            gc.collect()
            if len(hdd_neg)!=0 and len(hdd_pos)!=0:
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
                
                pos_train, pos_test = train_test_split(hdd_pos, test_size=0.2)
                
                #print(len(pos_train))
                #print(len(pos_test))
                
                neg_train, neg_test = train_test_split(hdd_neg, test_size=0.2)
                #print(len(neg_train))
                #print(len(neg_test))
                
                train_merged = [pos_train, neg_train]
                train_dataset = pd.concat(train_merged)
                # print(Counter(X_train))
                #print(train_dataset.describe)
                
                test_merged = [pos_test, neg_test]
                test_dataset = pd.concat(test_merged)
                # print(Counter(X_train))
                #print(test_dataset.describe)
                
                train_dataset = train_dataset.dropna()
                #print(train_dataset.describe)
                
                x = train_dataset.iloc[:, :-1].values
                y = train_dataset.iloc[:, -1].values
                
                X_resampled, y_resampled = SMOTE().fit_resample(x, y)
                #print(sorted(Counter(y_resampled).items()))
                #print(len(y_resampled))
                
                clf=RandomForestClassifier(n_estimators=100)
                clf.fit(X_resampled,y_resampled)
                
#                from sklearn import tree
#                clf = tree.DecisionTreeClassifier()
#                clf.fit(X_resampled,y_resampled)
                
                test_dataset = test_dataset.dropna()
                #print(test_dataset.describe)
                
                X_test = test_dataset.iloc[:, :-1].values
                y_test = test_dataset.iloc[:, -1].values
                
                y_pred=clf.predict(X_test)
                preds = clf.predict_proba(X_test)
                
                prob_score = calculate_accuracy(y_pred,y_test,preds)
                
                out_file.write("\n\n"+"model: "+model_name+"\n")
                out_file.write("\n")
                out_file.write("Accuracy: "+ str(metrics.accuracy_score(y_test, y_pred))+"\n")
                out_file.write("Recall: "+ str(metrics.recall_score(y_test, y_pred))+"\n")
                
#                print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
#                print("Recall: ",metrics.recall_score(y_test, y_pred))
                conf_matrix = metrics.confusion_matrix(y_test, y_pred)
                out_file.write(str(conf_matrix))
                out_file.write("\n")
                out_file.write(str(perf_measure(y_test, y_pred)))
                out_file.write("\n")
                out_file.write(prob_score)
                out_file.flush()
                
                from joblib import dump, load
                dump(clf, '../ml_models/'+model_name+'_'+day+'_rf.joblib') 
#                print(conf_matrix)
#                print(perf_measure(y_test, y_pred))
                
#                import seaborn as sn
#                import pandas as pd
#                import matplotlib.pyplot as plt
#                
#                df_cm = pd.DataFrame(conf_matrix, range(2), range(2))
#                plt.figure(figsize = (12,8))
#                sn.set(font_scale=2)#for label size
#                sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 28})# font size
#                
#                plt.savefig("../final_result/"+str(day)+"/"+model_name+"_rf.png")
                
                del hdd_pos
                del hdd_neg
                del train_dataset
                del test_dataset
                gc.collect()
#                count+=1
#                if count==2:
#                    break
        except:
            print(x)
            traceback.print_exc()