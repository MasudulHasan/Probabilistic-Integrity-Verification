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

selected_models = ['ST12000NM0007','ST4000DM000','ST8000DM002','ST8000NM0055']
# selected_models = ['ST8000DM002']
for model in selected_models:
    parent_folder = "/home/masudulhasanmasudb/Music/transfer_learning_result/"

    x_list=[]
    y_list=[]
    y_values=[]
    for month in range(1,10):
        file_list=glob.glob(parent_folder+str(month)+"/*")
        # print(file_list)
        x_list.append(month)
        y=[]
        for file_ in file_list:
            if model in file_:
                with open(file_) as in_file:
                    for line in in_file:
                        if "Recall" in line:
                            colon_index = line.rfind(":")
                            val = float("%.2f"%float(line[colon_index+1:].strip()))
                            # print(val)
                            y.append(val)
        y_list.append(y)

    for x in range(4):
        tmp=[]
        for y in range(9):
            tmp.append(y_list[y][x])
        print(tmp)
        y_values.append(tmp)

    
    legend= ['ST12000NM0007','ST4000DM000','ST8000DM002','ST8000NM0055']
    marker=['P','v','s','o','H']
    import seaborn as sns
    sns.set()
    sns.set_style("whitegrid")
    sns.set_palette("Set2")
    colours=['r','g','b','k','y']
    font_size=30
    for i in range(4):
        ax = sns.lineplot(x_list,y_values[i], marker=marker[i], label=legend[i],ms=18, lw=5)
    # ax = sns.lineplot(a,f_list[1][1:], marker='v', label=legend[1],ms=18, lw=5,)
    # ax = sns.lineplot(a,f_list[2][1:], marker='s', label=legend[2],ms=18, lw=5,)
    # ax = sns.lineplot(a,f_list[3][1:], marker='o', label=legend[3],ms=18, lw=5,)
    # ax = sns.lineplot(x_list[1:],f_list[4][1:], marker='H', label=legend[4],ms=18, lw=4,)
    plt.legend(markerscale=1, loc="upper center")
    plt.ylabel("Checksum Speedup", fontsize=font_size)
    plt.xlabel("File Size", fontsize=font_size)
    # plt.ylim(0,15)
    # plt.xlim(0,4)
    # ax.set_xticks(a)
    ax.set_xticklabels(x_list[1:], rotation=0)
    plt.setp(ax.get_legend().get_texts(), fontsize=font_size)
    ax.tick_params(axis='both', which='major')
    ax.tick_params(axis='y', which='major', labelsize=font_size)
    ax.tick_params(axis='x', which='major', labelsize=font_size)
    # plt.rcParams["figure.figsize"] = (12, 8)
    plt.tight_layout()
    # plt.savefig("/home/masudulhasanmasudb/Music/hdd_data/Result/transfer_learning_result/"+model+".pdf")
    plt.show()