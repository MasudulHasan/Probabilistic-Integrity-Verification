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

# with open("/home/masudulhasanmasudb/eclipse-workspace/Proactive_error_detection/file_size_1.txt") as in_file:
#     for line in in_file:
#         print(line)

lst =[]
t_count= 0
count =0
with open("/home/masudulhasanmasudb/eclipse-workspace/Proactive_error_detection/workload_1.txt") as in_file:
    for line in in_file:
        parts = line.strip().split(" ")
        print(parts)
        if int(parts[3])!=0:
            # lst.append(int(parts[3])/1000000000000)
            # count+=int(parts[3])/1000000000000
            # t_count+=1

            lst.append(int(parts[3]))
            count+=int(parts[3])
            t_count+=1

plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams.update({'font.size': 24})

x = np.arange(len(lst))
fig, ax = plt.subplots()
ax.plot(x,lst, marker='o', label="I_O_size")
#ax.axhline(y=average, xmin=0.0, xmax=1.0, color='r', label="Average Reacll value")
ax.set(ylabel='I/O size(TB)', xlabel='Date')
# ax.set_xticks(np.arange(0, 104, 4))
# ax.set_xticklabels(sorted(th_mp), rotation=90)
ax.grid()
ax.legend()
fig.tight_layout()
plt.savefig("/home/masudulhasanmasudb/Music/hdd_data/workload_per_day.png")
plt.show()

print("Avg ", count/t_count)