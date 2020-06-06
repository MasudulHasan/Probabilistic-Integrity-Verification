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

number_of_days = 1
lck = threading.Lock()

map_list = []
index_map ={}
date_dict={}
now = time.time()
count=-1
file_name=""
month = int(sys.argv[1])

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
        
    return final_prob+zero_error_prob
                

def get_best_disk(prob_list, n, threshold, number_of_error):
    alpha = .9
    min_cost = 999999
    from itertools import combinations, chain
#    n = len(serial_number_list)
    allsubsets = lambda n: list(chain(*[combinations(range(n), ni) for ni in range(n+1)]))    
    selected_subset=[]
    all_list=[i for i in range(n)]
    start = time.time()
    for x in allsubsets(n):
#        start = time.time()
        print(x)
        temp_list=[]
        for y in x:
            temp_list.append(y)
        
        un_selected_list=(list(set(all_list) - set(temp_list)))
#        print("un_selected_list "+str(un_selected_list))
        unslected_prob_list = []
        for i in un_selected_list:
            unslected_prob_list.append(prob_list[i])
            
        prob = 1.00
        for index in range(len(unslected_prob_list)):
            prob*=(1-unslected_prob_list[index])
        zero_error_prob = prob
        if(number_of_error==1):
            for index in range(len(unslected_prob_list)):
                prob+=((zero_error_prob/1-unslected_prob_list[index])*unslected_prob_list[index])
        
#        prob = 1 - calculate_error_prob(unslected_prob_list, number_of_error)
        prob=1-prob
        print("real prob ",prob)
        if(prob<=threshold):
            data_size = len(x)/n
            cost = (alpha*data_size)+(1-alpha)*prob
            print("prob ",prob," cost "+str(cost))
            if(cost<min_cost):
                min_cost = cost
                selected_subset = temp_list
                
#    print("after prob ",time.time()-start)
    return selected_subset                
    

def get_best_disk_greedy(prob_list, n, threshold, number_of_error):
    print("Greedy")
    alpha = 1
    
    new_prob_list=[x*alpha for x in prob_list]
    data_dict = defaultdict(list)
    for x in range(len(new_prob_list)):     
        data_dict[new_prob_list[x]].append(x)

    # print(data_dict)

    calculate_error_prob(new_prob_list,number_of_error)
    prob=1-calculate_error_prob(new_prob_list,number_of_error)
    print("greedy no disk prob ",prob)
    if(prob<=threshold):
        return []
    
    new_prob_list.sort(reverse = True)
    for x in range(len(new_prob_list)):
        prob=1-calculate_error_prob(new_prob_list[x+1:],number_of_error)
        print("x ",x," prob ",prob)
        if(prob<=threshold):
            selected_list=[]
            selected_probs=new_prob_list[:x+1]
            for number in selected_probs:
                # print(number)
                # print(data_dict)
                selected_list.append(data_dict[number][0])
                data_dict[number].remove(data_dict[number][0])
            return selected_list

        
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
def runner(number_of_max_disk, df, date,year,month,day):
    output_str =""
    for iter_n in range(100):
        try:
            #s = np.random.uniform(0,1)
        #   if s>.9:
            file_size = np.random.poisson(lam=1.165580e+07)
        #    else:
        #        file_size = np.random.poisson(lam=2.082032e+01)
            print(file_size)
            output_str += "\n\nfile size "+ str(file_size)+"\n"
            disk_number = int((file_size/1024)/50)+1
            if(disk_number>number_of_max_disk):
                disk_number = number_of_max_disk
            shape = df.shape
            if(disk_number> shape[0]):
                selected_disk = df
                
            else:
                selected_disk = df.sample(disk_number)
            start = time.time()

            # print("shape ",selected_disk.shape)

            pred_value = selected_disk[0].tolist()
            preds = selected_disk[2].tolist()
            next_day_label = selected_disk[3].tolist()
            
            # del df
            del selected_disk
            gc.collect()

            n = len(pred_value)
            threshold_index = randint(0, 44)
            number_of_error = randint(0, 1)
    #        number_of_error = 0
#                print("1 ",time.time()-start)
            start = time.time()
            selected_disk = get_best_disk_greedy(preds, n, thresholds[threshold_index], number_of_error)
            selected_disk_brute = get_best_disk(preds, n, thresholds[threshold_index], number_of_error)
            # print(selected_disk)
            # print(selected_disk_greedy)

#                print("2 ",time.time()-start, number_of_error)
            all_list=[i for i in range(n)]
            un_selected_list=(list(set(all_list) - set(selected_disk)))
            
            output_str+="greedy: \n"
            output_str+=str(selected_disk) + "\n"
            output_str+="brute: \n"
            output_str+=str(selected_disk_brute) + "\n"
            output_str+=str(date)+"\n"
            output_str+="predicted_value: \n"
            output_str+=str(pred_value) + "\n"
            output_str+="prob score: \n"
            output_str+=str(preds) + "\n"
            output_str+="threshold: "+str(thresholds[threshold_index])+"\n"
            output_str+="number of error : "+str(number_of_error)+"\n"
            print("greedy ",selected_disk)
            print(" brute ",selected_disk_brute)
            output_str+="next day real value: \n"
            output_str+=str(next_day_label)+"\n"
    
            output_str+=calculate_accuracy(pred_value, next_day_label, un_selected_list)
            output_str+="I/O saved = "+str((len(un_selected_list)/n)*100)+"\n"
            # output_str+="greedy: \n"
            # output_str+=str(selected_disk) + "\n"
            # output_str+="brute: \n"
            # output_str+=str(selected_disk_brute) + "\n"
        except:
            traceback.print_exc()
    return output_str


disk_model_name = 'ST4000DM000'
thresholds = np.arange(start=0.28, stop=.505, step=.005)
start_day=1
end_day =2
number_of_max_disk = 5
year = 2019
for part in range(1):
    stripe_size = 50
    if start_day==1:
        output_file = open("../result_log/greedy_approach/"+str(disk_model_name)+"_"+str(month)+"_first.txt","a+")
    else:
        output_file = open("../result_log/greedy_approach/"+str(disk_model_name)+"_"+str(month)+"_second.txt","a+")
    
    for day in range(start_day,end_day):
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
                df = pd.read_csv("../predicted_result_with_all/"+date+".csv", header=None)
                print(df.describe())
                shape = df.shape
                if shape[0]!=0: 
                    # wl = 2*np.random.poisson(lam=1.123983e+05)
                    wl =320000
                    print(wl)
                    output_str =""
                    for numof_iter in range(int(wl/1600)):
                        start = time.time()
                        pool = multiprocessing.Pool(processes=8)
                        outputs = [pool.apply_async(runner, args = (number_of_max_disk, df, date,year,month,day,)) for x in range(16)]
                        pool.close()
                        pool.join()
                        
                        print("duration =", time.time() - start)
                        output = [p.get() for p in outputs]

                        print(str(numof_iter)+ " done")
                        for s in output:
                            output_file.write(s)
                            output_file.flush()
                    
                    out_contents=[]
                        
                del(df)
                gc.collect()
            except:
                traceback.print_exc()
    
    output_file.close()
    start_day+=15
    end_day+=16
                    
                      
