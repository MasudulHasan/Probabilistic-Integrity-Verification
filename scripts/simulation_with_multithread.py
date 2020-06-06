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

number_of_days = 1
lck = threading.Lock()

map_list = []
index_map ={}
date_dict={}
now = time.time()
count=-1
file_name=""
with open("../map_2019.txt","r")as in_file:
    for line in in_file:
        if(len(line.strip())>0):
            if".csv" in line:
                if(count!=-1):
                    map_list.append(date_dict)
                    index_map[file_name]=count
                
#                date_dict.clear()
                date_dict={}
                count+=1
                file_name=line.strip()
            else:
                parts = line.strip().split(" ")
                date_dict[parts[0]] = int(float(parts[1]))
                               
print(count)

def get_lable(serial_number_list,date,year,month,day):
    global parent_folder_name
    next_day_label =[]
    
    now = datetime.datetime(year,month,day)
    next_day = (now + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    
    for x in serial_number_list:
        index = index_map[x+".csv"]
#        global lck
#        lck.acquire()
        current_value = map_list[index][date]
        next_day_value = map_list[index][next_day]
#        lck.release()
        
        if(current_value==next_day_value):
            next_day_label.append(0)
        else:
            next_day_label.append(1)
        
    return next_day_label
        

def Sort_Tuple(tup):   
    return(sorted(tup, key = lambda x: x[0], reverse = True))

def calculate_error_prob(prob_list, number_of_error):
#    print("prob list "+str(prob_list))
    zero_error_prob = 1.00
    for x in range(len(prob_list)):
        zero_error_prob*=(1-prob_list[x])
    
#    print("prob1 "+str(zero_error_prob))
    if(number_of_error==0):
        return zero_error_prob
    
    final_prob =0
    for x in range(len(prob_list)):
        final_prob+=((zero_error_prob/1-prob_list[x])*prob_list[x])
        
        
#        i = prob_list[x]
#        for y in range(len(prob_list)):
#            if x!=y:
#                i=i*(1-prob_list[y])
#        final_prob+=i
#    print(final_prob)
    return final_prob+zero_error_prob
                
                
    

def get_best_disk(prob_list, n, threshold, number_of_error):
#    print("threshold "+str(threshold))
#    print("prob: "+str(prob_list))
    
    
    
    alpha = .9
    min_cost = 999999
    from itertools import combinations, chain
#    n = len(serial_number_list)
    allsubsets = lambda n: list(chain(*[combinations(range(n), ni) for ni in range(n+1)]))
#    print("after lamda ",len(allsubsets))
    
    
    selected_subset=[]
    all_list=[i for i in range(n)]
    start = time.time()
    for x in allsubsets(n):
#        start = time.time()
#        print(x)
        temp_list=[]
        for y in x:
            temp_list.append(y)
        
        
        un_selected_list=(list(set(all_list) - set(temp_list)))
#        print("un_selected_list "+str(un_selected_list))
        unslected_prob_list = []
        for i in un_selected_list:
            unslected_prob_list.append(prob_list[:,1][i])
            
        prob = 1.00
        for index in range(len(unslected_prob_list)):
            prob*=(1-unslected_prob_list[index])
        zero_error_prob = prob
        if(number_of_error==1):
            for index in range(len(unslected_prob_list)):
                prob+=((zero_error_prob/1-unslected_prob_list[index])*unslected_prob_list[index])
        
#        prob = 1 - calculate_error_prob(unslected_prob_list, number_of_error)
        prob=1-prob
#        print("prob ",type(prob), " threshold ", type(threshold))
        if(prob<=threshold):
            data_size = len(x)/n
            cost = (alpha*data_size)+(1-alpha)*prob
#            print("cost "+str(cost))
            if(cost<min_cost):
                min_cost = cost
                selected_subset = temp_list
                
#    print("after prob ",time.time()-start)
    return selected_subset
            
        
def calculate_accuracy(pred_list, real_list, un_selected_list):
    tp=0 
    tn=0
    fp=0
    fn=0
    final_string=""
    for x in range(len(real_list)):
        if x in un_selected_list:
            if pred_list[x]== 1 and real_list[x]==1:
                tp+=1
            elif pred_list[x]== 0 and real_list[x]==1:
                fn+=1
            elif pred_list[x]== 0 and real_list[x]==0:
                tn+=1
            elif pred_list[x]== 1 and real_list[x]==0:
                fp+=1
            
    final_string+="TP, FP, TN, FN = "+str(tp)+" "+str(fp)+" "+str(tn)+" "+str(fn)+"\n"
    try:
        final_string+="Recall: "+ str(tp/(tp+fn))+"\n"
    except:
        final_string+="Recall: "+ str(0)+"\n"
    try:
        final_string+="extra: "+ str((fp/(tn+fp))*100)+"\n"
    except:
        final_string+="extra: "+ str(0)+"\n"
    return final_string
             
#selected_models = ['ST4000DM000', 'ST8000DM002', 'ST12000NM0007', 'ST8000NM0055', 'ST3000DM001', 'ST4000DX000']
selected_models = ['ST4000DM000']
out_contents = []
output = multiprocessing.Queue()
def runner(number_of_max_disk,df, date, year,month,day):
    output_str =""
    for iter_n in range(100):
        try:
            #s = np.random.uniform(0,1)
        #   if s>.9:
            file_size = np.random.poisson(lam=1.165580e+07)
        #    else:
        #        file_size = np.random.poisson(lam=2.082032e+01)
            
            output_str += "\n\nfile size "+ str(file_size)+"\n"
            disk_number = int((file_size/1024)/50)+1
            if(disk_number>number_of_max_disk):
                disk_number = number_of_max_disk
            
#            print(disk_number)
            if(disk_number> shape[0]):
                selected_disk = df
                
            else:
                selected_disk = df.sample(disk_number)
                
            serial_number = selected_disk.iloc[:, 1].values
            start = time.time()
            if len(serial_number)<=number_of_max_disk:
                selected_disk = selected_disk[columns_specified]
                pred_value = clf.predict(selected_disk)
                preds = clf.predict_proba(selected_disk)
                
                n = len(serial_number)
                threshold_index = randint(0, 100)
                number_of_error = randint(0, 1)
        #        number_of_error = 0
#                print("1 ",time.time()-start)
                start = time.time()
                selected_disk = get_best_disk(preds, n, thresholds[threshold_index], number_of_error)
#                print("2 ",time.time()-start, number_of_error)
                all_list=[i for i in range(n)]
                un_selected_list=(list(set(all_list) - set(selected_disk)))
                
                
                output_str+=str(date)+"\n"
                output_str+="predicted_value: \n"
                output_str+=str(pred_value) + "\n"
                output_str+="prob score: \n"
                output_str+=str(preds) + "\n"
                output_str+="threshold: "+str(thresholds[threshold_index])+"\n"
                output_str+="number of error : "+str(number_of_error)+"\n"
                
                next_day_label = get_lable(serial_number,date,year,month,day)
                output_str+="next day real value: \n"
                output_str+=str(next_day_label)+"\n"
        
                output_str+=calculate_accuracy(pred_value, next_day_label, un_selected_list)
                output_str+="I/O saved = "+str((len(un_selected_list)/n)*100)+"\n"
#                output.put(output_str)
                
    #            global lck
    #            lck.acquire()
    #            global out_contents
    #            out_contents.append(output_str)
    #            lck.release()
#                return output_str
        except:
            #                            print("error")
            traceback.print_exc()
#    print(output_str)
    return output_str



for disk_model_name in selected_models:

    hdd = pd.read_csv("../final_dataset/"+str(number_of_days)+'/'+str(disk_model_name)+'.csv', header=None)
    #hdd = pd.read_csv("../dataset_1.csv")
    hdd = hdd.drop(hdd.columns[6], axis=1)
    hdd = hdd.drop(hdd.columns[9], axis=1)
    hdd = hdd.drop(hdd.columns[14], axis=1)
    hdd = hdd.drop(hdd.columns[13], axis=1)
    hdd = hdd.dropna()
#    print(hdd.head())
    
    hdd_extra = pd.read_csv("../2019_files/"+str(number_of_days)+'/'+str(disk_model_name)+'.csv', header=None)
#    print(hdd_extra.head())
    hdd_extra = hdd_extra.drop(hdd_extra.columns[6], axis=1)
    hdd_extra = hdd_extra.drop(hdd_extra.columns[9], axis=1)
    hdd_extra = hdd_extra.drop(hdd_extra.columns[14], axis=1)
    hdd_extra = hdd_extra.drop(hdd_extra.columns[13], axis=1)
    hdd_extra = hdd_extra.dropna()
    
    hdd_merged = [hdd, hdd_extra]
    result = pd.concat(hdd_merged)
#    print(result.head())
#    result = result.dropna()
    
#    result = hdd
    
    x = result.iloc[:, :-1].values
    y = result.iloc[:, -1].values
    
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler()
    
    del hdd
    del hdd_extra
    del hdd_merged
    del result
    gc.collect()
    
    X_resampled, y_resampled = rus.fit_resample(x, y)
    print(Counter(y_resampled))
#    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)
    clf=RandomForestClassifier()
    clf.fit(X_resampled,y_resampled)
    
    
    features = [1, 4, 5, 7, 9, 12, 188, 193, 194, 197, 198, 199]
    columns_specified = []
    for feature in features:
        	columns_specified += ["smart_{0}_raw".format(feature)]
    
    stripe_size = 50
    
    thresholds = np.arange(start=0, stop=.505, step=.005)
    
    output_file = open("../result_log/model_simulation_new/"+str(disk_model_name)+"_july_first.txt","a+")
#    output_file = "../result_log/model_simulation_new/"+str(disk_model_name)+"_july_second.txt"
    number_of_max_disk = 10
    ##
    year = 2019
    end_year = 2019
    
    num_cpus = psutil.cpu_count(logical=False)
#    print(num_cpus)

    while year<=end_year:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        for month in range(7,8):
            for day in range(1,16):
                if month<=9:
                        month_str = "0"+str(month)
                else:
                    month_str = str(month)
                
                if day<=9:
                    day_str = "0"+str(day)
                else:
                    day_str = str(day)
                
                correctDate = None
                try:
                    newDate = datetime.datetime(year,month,day)
                    correctDate = True
                except ValueError:
                    correctDate = False
                if correctDate==True:
                    try:
                        date = str(year)+"-"+month_str+"-"+day_str
                        print(date)
                        df = pd.read_csv("../data/"+date+".csv")
    #                    df = df.loc[df['model'] == 'ST4000DM000']
                        df = df.loc[df['model'] == str(disk_model_name)]
                        df = df.loc[df['serial_number'] !="S300XQ5W"]
                        df = df.loc[df['serial_number'] !="W0Q7D8BD"]
                        df = df.loc[df['serial_number'] !="W3004WHH"]
                        shape = df.shape
                        if shape[0]!=0:
                            total_disk_number = 0
                            total_check_disk = 0
                            wl = 2*np.random.poisson(lam=1.123983e+05)
#                            wl =1
                            print(wl)
                            output_str =""
                            for numof_iter in range(int(wl/3200)):
#                                threads = []
#                                for i in range(0, 60):
                                
                                start = time.time()
                                pool = multiprocessing.Pool(processes=16)
                                outputs = [pool.apply_async(runner, args = (number_of_max_disk, df,date,year,month,day,)) for x in range(32)]
                                pool.close()
                                pool.join()
                                
                                print("done")
                                print("duration =", time.time() - start)
                                
#                                print("Output: {}".format(outputs)) 
                                output = [p.get() for p in outputs]
                                print(output)
                                
#                                results = [output.get() for p in outputs]
#                                print(results)
                                
                               
                                
#                                start = time.time()
#                                pool = multiprocessing.Pool() 
#                                pool = multiprocessing.Pool(processes=32) 
#                                processes = [multiprocessing.Process(target = runner, args = (number_of_max_disk,df, output_file, )) for x in range(32)]
#                                
#                                outputs = [pool.apply(runner, args = (number_of_max_disk,df, )) for x in range(num_cpus)]
##                                print("Input: {}".format(inputs)) 
##                                print("Output: {}".format(outputs))
##                                runner(number_of_max_disk, df, date, year,month,day)
#                                print("duration =", time.time() - start)
##                                pool.close()

                                    
                                    
#                                    t = threading.Thread(target = runner, args = (number_of_max_disk,df, output_file, ))
#                                    threads.append(t)
#                                    t.start()
#                                
#                                for t in threads:
#                                    t.join()
                                
                                print(str(numof_iter)+ " done")
#                                global out_contents
                                for s in output:
                                    output_file.write(s)
                                    output_file.flush()
                            
                            out_contents=[]
                                
        
#                            output_file.write(output_str)
#                            output_file.flush()
                        del(df)
                        gc.collect()
                    except:
                        traceback.print_exc()
                    
    
        year+=1                   
