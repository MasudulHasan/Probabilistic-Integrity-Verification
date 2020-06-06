import time
import glob
import random
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
import sys
import traceback
import threading
import datetime
from random import *
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

read_byte=0
count=0
line_list=[]
y_list=[]
with open("/home/masudulhasanmasudb/Music/b.txt") as in_file:
    for line in in_file:
        if "PM" in line:
            line_list.append(line.strip())

line_list = line_list[1:]
for line in line_list:
    # print(line)
    parts=line.split(" ")
    # print(parts)
    tmp =[]
    for x in parts:
        if len(x.strip())>0:
            tmp.append(x)
    # print(tmp)
    val = int(float(tmp[5]))
    # print(val)
    y_list.append(val)
    read_byte+=val
    # break

# print(read_byte/(1024*1024*1024))
print((read_byte*512)/(1024*1024*1024))
# print(count)

# legend= ['Se','ST4000DM000','ST8000DM002','ST8000NM0055']
marker=['P','v','s','o','H']
import seaborn as sns
sns.set()
sns.set_style("whitegrid")
sns.set_palette("Set2")
colours=['r','g','b','k','y']
font_size=30
a = np.arange(len(y_list))
# for i in range(4):
#     ax = sns.lineplot(x_list,y_values[i], marker=marker[i], label=legend[i],ms=18, lw=5)
ax = sns.lineplot(a,y_list, marker='v',ms=18, lw=5)
# ax = sns.lineplot(a,f_list[2][1:], marker='s', label=legend[2],ms=18, lw=5,)
# ax = sns.lineplot(a,f_list[3][1:], marker='o', label=legend[3],ms=18, lw=5,)
# ax = sns.lineplot(x_list[1:],f_list[4][1:], marker='H', label=legend[4],ms=18, lw=4,)
plt.legend(markerscale=1, loc="upper center")
plt.ylabel("Checksum Speedup", fontsize=font_size)
plt.xlabel("File Size", fontsize=font_size)
# plt.ylim(0,15)
# plt.xlim(0,4)
# ax.set_xticks(a)
# ax.set_xticklabels(x_list[1:], rotation=0)
plt.setp(ax.get_legend().get_texts(), fontsize=font_size)
ax.tick_params(axis='both', which='major')
ax.tick_params(axis='y', which='major', labelsize=font_size)
ax.tick_params(axis='x', which='major', labelsize=font_size)
# plt.rcParams["figure.figsize"] = (12, 8)
plt.tight_layout()
# plt.savefig("/home/masudulhasanmasudb/Music/hdd_data/Result/transfer_learning_result/"+model+".pdf")
plt.show()