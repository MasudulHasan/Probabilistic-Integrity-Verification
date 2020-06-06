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

# parent_folder_name = "../predicted_result_with_all/"
# folder_list=glob.glob(parent_folder_name+"*")

# out_file = open("../error_each_day.txt","a+")
# year = 2019
# for month in range(1,10):
#     for day in range(1,32):
#         if month<=9:
#                 month_str = "0"+str(month)
#         else:
#             month_str = str(month)
        
#         if day<=9:
#             day_str = "0"+str(day)
#         else:
#             day_str = str(day)
#         try:
#             date = str(year)+"-"+month_str+"-"+day_str
#             # slash_index = file_.rfind("/")
#             # dot_index = file_.rfind(".csv")

#             # file_name = file_[slash_index+1:dot_index]
#             # print(file_name)
#             count =0 
#             df = pd.read_csv("../predicted_result_with_all/"+date+".csv", header=None)
#             next_day_label = df[3].tolist()
#             for x in next_day_label:
#                 if x==1:
#                     count+=1
#             print(count)
#             out_file.write(str(date)+" "+str(count)+"\n")
#         except:
#             pass
#             # break

lst =[]
t_count= 0
count =0
with open("../error_each_day.txt","r+") as in_file:
    for line in in_file:
        parts = line.strip().split(" ")
        print(parts)
        lst.append(int(parts[1]))
        count+=int(parts[1])
        t_count+=1

print(t_count)
print(count)

plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams.update({'font.size': 24})

x = np.arange(len(lst))
fig, ax = plt.subplots()
ax.plot(x,lst, marker='o', label="E")
#ax.axhline(y=average, xmin=0.0, xmax=1.0, color='r', label="Average Reacll value")
ax.set(ylabel='#error', xlabel='Date')
# ax.set_xticks(np.arange(0, 104, 4))
# ax.set_xticklabels(sorted(th_mp), rotation=90)
ax.grid()
# ax.legend()
fig.tight_layout()
plt.savefig("/home/masudulhasanmasudb/Music/hdd_data/error_per_day.png")
plt.show()

print("Avg ", count/t_count)