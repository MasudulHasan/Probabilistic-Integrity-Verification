#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 01:55:10 2020

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

parent_folder_name = "../throughput_comparison/"
folder_list=glob.glob(parent_folder_name+"*")
#print(folder_list)

values=[]

for x in folder_list:
    if "comet" not in x:
        value_list =[]
        with open(x,"r+") as in_file:
            for line in in_file:
                if len(line.strip())>0:
                    if "MB" not in line and "GB" not in line:
        #                print(line)
                        value_list.append(float(line.strip()))
            
            i=0
    #        print(value_list)
            while(i<len(value_list)):
                values.append(value_list[i+1]/value_list[i])
                print(str(value_list[i+1])+" "+str(value_list[i])+" "+str(value_list[i+1]/value_list[i]))
                i+=2
        
#        print(x)
#        print(values)

#print(values)
        
print(max(values))
print(min(values))
print(np.average(values))