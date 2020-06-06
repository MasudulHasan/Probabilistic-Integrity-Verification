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

# df = pd.read_csv("/home/masudulhasanmasudb/Music/hdd_data/predicted_result_with_all/2019-01-01.csv", header=None)
# print(df)
# df = df[df[0]!=df[3]]
# print(df)
# df = df[df[0]==0]
# print(df)






# parent_folder_name = "/home/masudulhasanmasudb/Music/hdd_data/result_log/greedy_approach/"
parent_folder_name = "/home/masudulhasanmasudb/Music/hdd_data/result_log/model_simulation_new/"
folder_list=glob.glob(parent_folder_name+"*")
print(folder_list)
i=0

# for file_ in folder_list:
#     if "old" not in file_ and "local" not in file_:
#         with open(file_,"r+") as in_file:
#             for line in in_file:
#                 if(len(line.strip()))>0:
#                     line_list.append(line)

# file_ = parent_folder_name+"ST4000DM000_1_first.txt"

for file_ in folder_list:
    line_list=[]
    slash_index = file_.rfind("/")
    dot_index = file_.rfind(".txt")

    file_name = file_[slash_index+1:dot_index]


    with open(file_,"r+") as in_file:
        for line in in_file:
            if(len(line.strip()))>0:
                line_list.append(line)


    print(len(line_list))
    # print(line_list)
    th_mp={}
    i_o_mp={}
    c_mp={}
    for i in range (len(line_list)):
        if "threshold:" in line_list[i]:
            c_index = line_list[i+1].rfind(":")
            error_number = int(line_list[i+1][c_index+1:].strip())
                    
            if error_number==1:

                p_ = line_list[i+7].strip().split("=")
                i_o=float(p_[1].strip())

                col_index = line_list[i].rfind(":")
                th = float(line_list[i][col_index+1:])
                
                if th in i_o_mp:
                    val = i_o_mp[th]
                    i_o_mp[th] = val+i_o
                else:
                    i_o_mp[th] = i_o
                
                if th in c_mp:
                    val = c_mp[th]
                    c_mp[th] = val+1
                else:
                    c_mp[th] = 1

                # print(line_list[i+4])
                _index = line_list[i+4].rfind("=")
                parts_ = line_list[i+4][_index+1:].strip().split(" ")
                value=int(parts_[3].strip())
                # print(parts_)
                # if(value!=0):
                    
                    # if th==0.28:
                    # print("ERROR thresh ",line_list[i]," ",line_list[i +4])
                    # print(line_list[i-1]," ",line_list[i-3])
                    # print(line_list[i+3])
                    # print(line_list[i-6])
                # if len(line_list[i-8].strip()[1:-1])>0:
                #     print("selected ",len(line_list[i-8].strip()[1:-1]))
                # print(th)
                
                if th in th_mp:
                    val = th_mp[th]
                    th_mp[th] = value+val
                else:
                    th_mp[th] = value

    print(th_mp)

    error_number_list=[]
    count=0
    for k in sorted(th_mp):
        print(str(k)+" "+str((th_mp[k])))
        error_number_list.append(th_mp[k])
        count+=th_mp[k]

    print(count)

    values = []
    for k in sorted(i_o_mp):
        # print(str(key)+" "+str((threshold_value_map[key]/(count_map[key]*100))))
        values.append((i_o_mp[k]/(c_mp[k]*100)))


    plt.rcParams["figure.figsize"] = (20, 10)
    plt.rcParams.update({'font.size': 24})

    x = np.arange(len(values))
    fig, ax = plt.subplots()
    ax.plot(x,values, marker='o', label="I_O_saved")
    #ax.axhline(y=average, xmin=0.0, xmax=1.0, color='r', label="Average Reacll value")
    ax.set(ylabel='I/O saved(%)', xlabel='Thresholds')
    ax.set_xticks(np.arange(0, 104, 4))
    ax.set_xticklabels(sorted(th_mp), rotation=90)
    ax.grid()
    ax.legend()
    fig.tight_layout()
    # plt.savefig("/home/masudulhasanmasudb/Music/hdd_data/Result/i_o_zero_greedy.png")
    # plt.show()

    thresholds_list=np.arange(start=0, stop=.52, step=.02)
    x = np.arange(len(error_number_list))
    fig, ax = plt.subplots()
    ax.plot(x,error_number_list, marker='o', label="fn")
    #ax.axhline(y=average, xmin=0.0, xmax=1.0, color='r', label="Average Reacll value")
    ax.set(ylabel='Error missed', xlabel='Thresholds')
    ax.set_xticks(np.arange(0, 104, 4))
    ax.set_xticklabels(thresholds_list, rotation=90)
    ax.grid()
    ax.legend()
    fig.tight_layout()
    plt.savefig("/home/masudulhasanmasudb/Music/hdd_data/Result/"+file_name+"_brute.png")
    # plt.show()