#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:08:10 2019

@author: masudulhasanmasudb
"""

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
from subprocess import PIPE, Popen
#import seaborn as sns


parent_folder_name = "/home/masudulhasanmasudb/eclipse-workspace/Proactive_error_detection/output_files/"
folder_list=glob.glob(parent_folder_name+"*")
##out_file = open("done_so_far.txt","a+")
#print(len(folder_list))

#output_file = open("filtered_files.txt","a+")
#for x in folder_list:
##    print(x)
#    df = pd.read_csv(x)
#    model = df['model'].iloc[0]
#    if model.strip() =='ST4000DM000':
#        output_file.write(str(x)+"\n")
#        output_file.flush()
#    print(model)

with open("filtered_files.txt","r") as in_file:
    for line in in_file:
        proc = Popen(['mv', line.strip(),'output_files/'], universal_newlines=True, stdout=PIPE)
        res = proc.communicate()[0]
#        commad = ['mv', line.strip(),'output_files/']
        