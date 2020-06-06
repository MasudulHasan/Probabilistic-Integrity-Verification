#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 21:25:53 2020

@author: masudulhasanmasudb
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 11:56:22 2019

@author: masudulhasanmasudb
"""

import time,math
import glob,random
import datetime
import os
import subprocess
import shlex
import gc,sys, traceback
import pandas as pd
import collections
import matplotlib.pyplot as plt

#day = int(sys.argv[1])
day =1
#print(day)
def all_same(items):
    return all(x == 0 for x in items)

def get_label_list(df_list, day):
    label_list=[]
    for i in range(len(df_list)-1):
        if math.isnan(df_list[i+1]) or math.isnan(df_list[i]):
            label_list.append(0)
        else:
            if df_list[i+1]!= df_list[i]:
                label_list.append(1)
            else:
               label_list.append(0)

    return label_list
count=0
#for year in range(2013,20120):
#    parent_folder_name = "/home/masudulhasanmasudb/eclipse-workspace/Proactive_error_detection/output_files_"+str(year)+"/"
#    folder_list=glob.glob(parent_folder_name+"*")
#    for x in folder_list:
#        try:
#            df = pd.read_csv(x, header=None)
#            serial_number = df[0][1]
#            print(serial_number)
#            output_file = open("../combined_file/"+serial_number+".csv","a+")
#            df.to_csv(output_file, header=False, index=False)
#            
#            count+=1
#            print(count)
#            del df
#            gc.collect()
#        except:
#            print(x)
#            traceback.print_exc()

parent_folder_name = "../combined_file/"
folder_list=glob.glob(parent_folder_name+"*")
bit_error_final_map={}
sector_error_final_map={}
failue_map={}

for x in folder_list:
    try:
        df = pd.read_csv(x, header=None)
        df_list = df[6].tolist()
        label_list= get_label_list(df_list, day)
        
        bit_error_list = df[10].tolist()
        bit_error_labels= get_label_list(bit_error_list, day)
        
        failures = df[3].tolist()
        date_list=df[1].tolist()
        start_date=df[1][0]
        for i in range(len(failures)):
            date=date_list[i]
            hasFailed = failures[i]
            if hasFailed == 1:
                age=datetime.datetime.strptime(date, '%Y-%M-%d')-datetime.datetime.strptime(start_date, '%Y-%M-%d')
                if age.days in failue_map:
                    value = failue_map[age.days]
                    failue_map[age.days]=value+1
                else:
                    failue_map[age.days]=1
        
#        print(label_list)
        date_list=df[1][:-1]
        start_date=df[1][0]
        for i in range(len(label_list)):
            date=date_list[i]
            bit_error_value = bit_error_labels[i]
            sector_error_value = label_list[i]
            if sector_error_value==1:
                age=pd.to_datetime(date)-pd.to_datetime(start_date)
#                print(age)
                if age.days in sector_error_final_map:
                    value = sector_error_final_map[age.days]
                    sector_error_final_map[age.days]=value+1
                else:
                    sector_error_final_map[age.days]=1
                
                
            if bit_error_value==1:
                age=pd.to_datetime(date)-pd.to_datetime(start_date)
                if age.days in bit_error_final_map:
                    value = bit_error_final_map[age.days]
                    bit_error_final_map[age.days]=value+1
                else:
                    bit_error_final_map[age.days]=1
                
                
        count+=1
        print(count)
#        if count==100:
#            break
        del df
        gc.collect()
    except:
        print(x)
        traceback.print_exc()

#print(bit_error_final_map)
#print(sector_error_final_map)
#print(failue_map)

v = [i for i in sorted (sector_error_final_map.keys())]
x_list=[]
value_list=[]
for x in range(1,2363):
    if x in sector_error_final_map:
#        print(str(x)+" -> "+str(sector_error_final_map[x]))
        value_list.append(sector_error_final_map[x])
    else:
        value_list.append(0)
    x_list.append(x)
    
plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams.update({'font.size': 24})
fig, ax = plt.subplots()
ax.plot(x_list, value_list)

ax.set(xlabel='Age (day)', ylabel='#of error', title='Age of Disk When Sector Error Happen')
ax.grid()

fig.savefig("../sector_error_age.png")
#plt.show()

v = [i for i in sorted (bit_error_final_map.keys())]
x_list=[]
value_list=[]
for x in range(1,2363):
    if x in bit_error_final_map:
#        print(str(x)+" -> "+str(bit_error_final_map[x]))
        value_list.append(bit_error_final_map[x])
    else:
        value_list.append(0)
    x_list.append(x)
    
plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams.update({'font.size': 24})
fig, ax = plt.subplots()
ax.plot(x_list, value_list)

ax.set(xlabel='Age (day)', ylabel='#of error', title='Age of Disk When Bit Error Happen')
ax.grid()

fig.savefig("../bit_error_age.png")
#plt.show()

v = [i for i in sorted (failue_map.keys())]
x_list=[]
value_list=[]
for x in range(1,2363):
    if x in failue_map:
#        print(str(x)+" -> "+str(bit_error_final_map[x]))
        value_list.append(failue_map[x])
    else:
        value_list.append(0)
    x_list.append(x)
    
plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams.update({'font.size': 24})
fig, ax = plt.subplots()
ax.plot(x_list, value_list)

ax.set(xlabel='Age (day)', ylabel='#of error', title='Age of Disk When Disk Fails')
ax.grid()

fig.savefig("../failure_age.png")


#output_file = open("../bit_error_result.txt","a+")
#for (k,v) in bit_error_final_map.items():
##    print(str(k)+" : "+str(v))
#    output_file.write(str(k)+" : "+str(v)+"\n")
#    output_file.flush()
#
#output_file = open("../sector_error_result.txt","a+")
#for (k,v) in sector_error_final_map.items():
##    print(str(k)+" : "+str(v))
#    output_file.write(str(k)+" : "+str(v)+"\n")
#    output_file.flush()

#print(bit_error_final_map)
#print(sector_error_final_map)
