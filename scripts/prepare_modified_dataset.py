#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 01:14:31 2020

@author: masudulhasanmasudb
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 11:56:22 2019

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

#parent_folder_name = "/home/masudulhasanmasudb/eclipse-workspace/Proactive_error_detection/disk_model_files/"
parent_folder_name = "/home/masudulhasanmasudb/Downloads/data_Q3_2019/data_Q3_2019/"
folder_list=glob.glob(parent_folder_name+"*")
#print(folder_list)

features = [1, 4, 5, 7, 9, 12, 187, 188, 193, 194, 196, 197, 198, 199]
columns_specified = []
for feature in features:
    	columns_specified += ["smart_{0}_raw".format(feature)]
columns_specified = ["serial_number", "date", "model", "failure"] + columns_specified
print(columns_specified)
count = 0

for x in folder_list:
    try:
        s_index = x.rfind("/")
        file_name = x[s_index+1:]
        df = pd.read_csv(x)
        df = df[columns_specified]
        output_file = open("../modified_data/"+file_name,"a+")
        df.to_csv(output_file, header=False, index=False)
        count+=1
        print(count)
#        if(count==10):
#            break
        del df
        gc.collect()
    except:
        print(x)
        traceback.print_exc()

#list1=[columns_specified]
#df1 = pd.DataFrame(list1) 
#for x in folder_list:
#    s_index = x.rfind("/")
#    file_name = x[s_index+1:]
#    print(file_name)
#    df = pd.read_csv(x, header=None)
##    print(df)
#    df_result = pd.concat([df1, df]) 
#    df_result.reset_index(drop=True, inplace=False)
#    output_file = open("/home/masudulhasanmasudb/eclipse-workspace/Proactive_error_detection/modified_disk_model_files/"+file_name,"a+")
#    df_result.to_csv(output_file, header=False, index=False)
##    print(df_result)
#    
#    del df
#    del df_result
#    gc.collect()