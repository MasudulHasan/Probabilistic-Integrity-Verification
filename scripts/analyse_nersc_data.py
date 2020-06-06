#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:51:42 2020

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

parent_folder_name = "/home/masudulhasanmasudb/Downloads/Anonymized_ISDCT_Data/"
folder_list=glob.glob(parent_folder_name+"*")
#print(folder_list)

for x in folder_list:
    if x.endswith(".csv"):
#        print(x)
        df = pd.read_csv(x)
        for col in df.columns: 
            print(col)
        break
