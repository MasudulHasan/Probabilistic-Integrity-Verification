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
from scipy import stats

number_of_days = 1
lck = threading.Lock()

map_list = []
index_map ={}
date_dict={}
now = time.time()
count=-1
file_name=""
number_of_max_disk = int(sys.argv[1])
month = 8


def calculate_error_prob(prob_list, number_of_error,stripe_size):
    if len(prob_list)==0:
        return 0
    if len(prob_list)==1:
        return prob_list[0]
    zero_error_prob = 1.00
    for x in range(len(prob_list)):
        zero_error_prob*=(1-prob_list[x])
    # print(zero_error_prob)
    final_prob =0
    for x in range(len(prob_list)):
        final_prob+=((zero_error_prob/(1-prob_list[x]))*prob_list[x])
        # print(final_prob)
        
    return final_prob

def get_best_disk(prob_list, n, number_of_error,file_size,factor,block_size_list,alpha):
    normalised_prob_list = stats.zscore(prob_list).tolist()
    normalised_block_size_list = stats.zscore(block_size_list).tolist()
    threshold = pow(10, -15)/factor
    # alpha = .9
    min_cost = -1000000000
    from itertools import combinations, chain
#    n = len(serial_number_list)
    allsubsets = lambda n: list(chain(*[combinations(range(n), ni) for ni in range(n+1)]))    
    selected_subset=[]
    all_list=[i for i in range(n)]
    start = time.time()
    selected_data_size_list=[]
    selected_prob_list=[]
    selected_subset_list=[]
    for x in allsubsets(n):
        temp_list=[]
        data_size=0
        for y in x:
            data_size+=normalised_block_size_list[y]
            temp_list.append(y)
        un_selected_list=(list(set(all_list) - set(temp_list)))
        unslected_prob_list = []
        normalised_unslected_prob_list=[]
        for i in un_selected_list:
            unslected_prob_list.append(prob_list[i])
            normalised_unslected_prob_list.append(normalised_prob_list[i])
            
        stripe_size=0 #doesn't matter
        prob=calculate_error_prob(unslected_prob_list,number_of_error,stripe_size)
        val = prob/(file_size*8)
        # print("real prob ",prob)
        if val<=threshold or ((val-threshold)/threshold)<=.1:
            # n_prob=calculate_error_prob(normalised_unslected_prob_list,number_of_error,stripe_size)
            selected_data_size_list.append(data_size)
            selected_prob_list.append(prob)
            selected_subset_list.append(temp_list)
            # # data_size = len(x)/n
            # print(temp_list)
            # print(normalised_unslected_prob_list)
            # n_prob=calculate_error_prob(normalised_unslected_prob_list,number_of_error,stripe_size)
            # print(n_prob)
            # cost = (alpha*data_size)+(1-alpha)*n_prob
            # print("prob ",prob," cost "+str(cost))
            # if min_cost==-1000000000:
            #     min_cost = cost
            #     selected_subset = temp_list
            # elif(cost<min_cost):
            #     min_cost = cost
            #     selected_subset = temp_list
                
    # print("after prob ",time.time()-start)
    selected_subset=[]
    normalized_selected_prob_list = stats.zscore(selected_prob_list).tolist()
    for x in range(len(normalized_selected_prob_list)):
      cost = (alpha*selected_data_size_list[x])+(1-alpha)*normalized_selected_prob_list[x]
      if min_cost==-1000000000:
          min_cost = cost
          selected_subset = selected_subset_list[x]
      elif(cost<min_cost):
          min_cost = cost
          selected_subset = selected_subset_list[x]
    return selected_subset


def calibration(data, train_pop, target_pop, sampled_train_pop, sampled_target_pop):
    calibrated_data = ((data * (target_pop / train_pop) / (sampled_target_pop / sampled_train_pop)) / (( (1 - data) * (1 - target_pop / train_pop) / (1 - sampled_target_pop/sampled_train_pop)) + (data * (target_pop / train_pop) / (sampled_target_pop / sampled_train_pop))))
    return calibrated_data


class TNode:
    def __init__(self, cost, prob, file_size, position):
        self.cost = cost
        self.prob = prob
        self.file_size = file_size
        self.position = position

def cmp_func(a, b):
    if a.cost > b.cost:
        return 1
    else:
        return -1

def get_best_disk_greedy_new(prob_list, n, number_of_error,file_size_list,factor,alpha, file_size):
    
    normalised_file_size_list = stats.zscore(file_size_list).tolist()

    # print("prob ",normalised_prob_list)
    # print("file size ",normalised_file_size_list)

    threshold = pow(10, -15)/factor
    new_prob_list=[x for x in prob_list]

    selected_prob_list=[]
    for i in range(len(prob_list)):
        new_lst = prob_list[:i]+prob_list[i+1:]
        prob = calculate_error_prob(new_lst,number_of_error,stripe_size)
        selected_prob_list.append(prob)
      
    normalised_prob_list = stats.zscore(selected_prob_list).tolist()
    
    cost_list=[]
    for i in range(len(normalised_file_size_list)):
        cost=(alpha*normalised_file_size_list[i])+((1-alpha)*normalised_prob_list[i])
        # print(cost)
        p1 = TNode(cost, prob_list[i],file_size_list[i],i)
        # print(p1)
        cost_list.append(p1)
    # print(cost_list)

    # for x in cost_list:
      # print(x.cost)

    prob=calculate_error_prob(new_prob_list,number_of_error,stripe_size)
    val = prob/(file_size*8)
    if val<=threshold or ((val-threshold)/threshold)<=.1:
        return []
    
    from functools import cmp_to_key
    cmp_items_py3 = cmp_to_key(cmp_func)
    cost_list.sort(key = cmp_items_py3)
    # for x in cost_list:
    #   print("cost ",x.cost)

    selected_so_far=[]
    while(True):
        selected_so_far.append(cost_list[0])
        cost_list=cost_list[1:]
        new_prob_list=[]
        for cal in cost_list:
            new_prob_list.append(cal.prob)
        # print("new " ,new_prob_list)
        prob=calculate_error_prob(new_prob_list,number_of_error,stripe_size)
        val = prob/(file_size*8)
        # print("val ",val)
        if val<=threshold or ((val-threshold)/threshold)<=.1:
            selected_list=[]
            for number in selected_so_far:
                selected_list.append(number.position)
            return selected_list
        else:
            # cost_list=cost_list[1:]
            res_prob_list=[]
            res_file_size=[]
            positions=[]
            for cal in cost_list:
                res_prob_list.append(cal.prob)
                res_file_size.append(cal.file_size)
                positions.append(cal.position)

            res_normalised_file_size_list = stats.zscore(res_file_size).tolist()

            res_selected_prob_list=[]
            for i in range(len(res_prob_list)):
                new_lst = prob_list[:i]+res_prob_list[i+1:]
                prob = calculate_error_prob(new_lst,number_of_error,stripe_size)
                res_selected_prob_list.append(prob)
              
            res_normalised_prob_list = stats.zscore(res_selected_prob_list).tolist()
            
            res_cost_list=[]
            for i in range(len(res_normalised_file_size_list)):
                cost=(alpha*res_normalised_file_size_list[i])+((1-alpha)*res_normalised_prob_list[i])
                # print(cost)
                p1 = TNode(cost, res_prob_list[i],res_file_size[i],positions[i])
                # print(p1)
                res_cost_list.append(p1)

            cmp_items_py3 = cmp_to_key(cmp_func)
            res_cost_list.sort(key = cmp_items_py3)


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

def calculate_io_save(block_size_list, un_selected_list):
    saved_io=0
    total_io = 0
    for i in range(len(block_size_list)):
        if i in un_selected_list:
            saved_io+=block_size_list[i]
        total_io+=block_size_list[i]
    
    return saved_io, total_io, (saved_io/total_io)*100

             
#selected_models = ['ST4000DM000', 'ST8000DM002', 'ST12000NM0007', 'ST8000NM0055', 'ST3000DM001', 'ST4000DX000']
selected_models = ['ST12000NM0007']
out_contents = []
output = multiprocessing.Queue()
def runner(number_of_max_disk, df, date,year,month,day):
    output_str =""
    for iter_n in range(1):
        try:
            number_of_file = 4
            file_size_list=[]
            total_file_size=0
            for temp_v in range(number_of_file):
                s = np.random.uniform(0,1)
                # if s>.9:
                file_size = np.random.poisson(lam=1.165580e+07)
                # else:
                # file_size = np.random.poisson(lam=2.082032e+01)
                total_file_size+=file_size
                file_size_list.append(file_size)
            
            output_str += "\n\nfile size "+ str(file_size_list)+"\n"
            total_disk_number = number_of_max_disk
            # disk_number_list=[]
            block_size_list=[]
            disk_per_file = int(total_disk_number/number_of_file)
            
            for temp_v in range(number_of_file):
                stripe_size = int(file_size_list[temp_v]/disk_per_file)
                for block in range(disk_per_file):
                    block_size_list.append(stripe_size)

            # for temp_v in range(number_of_file):
            #     disk_number = min (number_of_max_disk, int(file_size_list[temp_v]/(128*1024))+1)
            #     total_disk_number+=disk_number
            #     stripe_size = int(file_size_list[temp_v]/disk_number)
            #     for block in range(disk_number):
            #         block_size_list.append(stripe_size)
                
            # print("total disk ",total_disk_number)
            shape = df.shape
            # if(disk_number> shape[0]):
            #     selected_disk = df
            #     disk_number = shape[0]
                
            # else:
            import random
            res = random.sample(range(0,  shape[0]-1), total_disk_number)
            # res = [random.randrange(0, shape[0]-1, 1) for i in range(disk_number)] 
            # selected_disk = df.sample(disk_number)
            selected_disk = df.iloc[res, :]
            start = time.time()

            # print("shape ",selected_disk.shape)

            pred_value = selected_disk[0].tolist()
            preds = selected_disk[2].tolist()
            next_day_label = selected_disk[3].tolist()
            
            # del df
            del selected_disk
            gc.collect()

            n = len(pred_value)
            # threshold_index = randint(0, 6)
            threshold_index = 2
            # number_of_error = randint(0, 1)
            number_of_error = 0
#                print("1 ",time.time()-start)
            start = time.time()
            alpha_index = randint(0, 10)
            # stripe_size = int(file_size/disk_number)
            
            if month == 8:
                t=[calibration(x, 15245762, 1306, 2612,1306 ) for x in preds]
            if month == 9:
                t=[calibration(x, 16321688, 1616, 3232,1616 ) for x in preds]
            prob_list=[]
            divisor = 16*pow(10,9)
            for val in range(len(t)):
                prob_list.append(float((t[val]*block_size_list[val])/divisor))


            # selected_disk = get_best_disk_greedy(prob_list, n, thresholds[threshold_index], number_of_error,stripe_size)
            selected_disk = get_best_disk_greedy_new(prob_list, n, number_of_error,block_size_list,10,alphas[alpha_index],total_file_size)
            #selected_disk_brute = get_best_disk(prob_list, n, number_of_error,total_file_size,10,block_size_list,alphas[alpha_index])

            # if selected_disk is not None:
            #     all_list=[i for i in range(n)]
            #     # print("all list", all_list)
            #     # print("selected ", selected_disk)
            #     # print("selected brute ", selected_disk_brute)
            #     un_selected_list=(list(set(all_list) - set(selected_disk)))
            #     output_str+="greedy: \n"
            #     output_str+=str(selected_disk) + "\n"
            #     output_str+="brute: \n"
            #     output_str+=str(selected_disk_brute) + "\n"
            #     output_str+="block size list: \n"
            #     output_str+=str(block_size_list) + "\n"
            #     output_str+="calculated prob score: \n"
            #     output_str+=str(prob_list) + "\n"
            #     output_str+=str(date)+"\n"
            #     output_str+="predicted_value: \n"
            #     output_str+=str(pred_value) + "\n"
            #     output_str+="prob score: \n"
            #     output_str+=str(preds) + "\n"
            #     output_str+="alpha: "+str(alphas[alpha_index])+"\n"
            #     output_str+="number of error : "+str(number_of_error)+"\n"
                
            #     output_str+="next day real value: \n"
            #     output_str+=str(next_day_label)+"\n"
        
            #     output_str+=calculate_accuracy(pred_value, next_day_label, un_selected_list)
            #     saved_io, total_io, perc = calculate_io_save(block_size_list, un_selected_list)
            #     output_str+="I/O saved = "+str(total_io)+" "+str(saved_io)+" "+str(perc)+"\n"

            #     un_selected_list_brute=(list(set(all_list) - set(selected_disk_brute)))
            #     output_str+="brute force result\n"
            #     output_str+=calculate_accuracy(pred_value, next_day_label, un_selected_list_brute)
            #     saved_io, total_io, perc = calculate_io_save(block_size_list, un_selected_list_brute)
            #     output_str+="brute force saved = "+str(total_io)+" "+str(saved_io)+" "+str(perc)+"\n"
        except:
            traceback.print_exc()
    return output_str

program_start = time.time()

disk_model_name = 'ST12000NM0007'
alphas = [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]
start_day=1
end_day =2
# number_of_max_disk = 8
year = 2019
for part in range(1):
    stripe_size = 50
    # if start_day==1:
    output_file = open("../result_log/test.txt","a+")
    # else:
    #     output_file = open("../result_log/greedy_vs_brute/"+str(disk_model_name)+"_"+str(month)+"_second_combined.txt","a+")
    
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
                # if disk_model_name =="ST12000NM0007":
                #     df = pd.read_csv("../predicted_result_new/"+date+".csv", header=None)
                # else:
                df = pd.read_csv("../predicted_result_"+disk_model_name+"/"+date+".csv", header=None)
                #print(df.describe())
                shape = df.shape
                if shape[0]!=0: 
                    # wl = int(np.random.poisson(lam=1.123983e+05)*0.3)
                    # wl = int(np.random.poisson(lam=1.123983e+05))
                    wl =3200*10
                    #print(wl)
                    output_str =""
                    for numof_iter in range(1):
                        start = time.time()
                        pool = multiprocessing.Pool(processes=1)
                        outputs = [pool.apply_async(runner, args = (number_of_max_disk, df, date,year,month,day,)) for x in range(1)]
                        pool.close()
                        pool.join()
                        
                        #print("duration =", time.time() - start)
                        output = [p.get() for p in outputs]

                        # print(str(numof_iter)+ " done")
                        for s in output:
                            output_file.write(s)
                            output_file.flush()
                    
                    out_contents=[]
                        
                del(df)
                gc.collect()
            except:
                traceback.print_exc()
    
    # output_file.close()
    start_day+=15
    end_day+=16
                    
                      
print("program duration =", time.time() - program_start , " for disk = ", number_of_max_disk)