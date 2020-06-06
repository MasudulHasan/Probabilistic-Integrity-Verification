#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:52:08 2019

@author: masudulhasanmasudb
"""

import time
import glob,random
import datetime
import os
import subprocess
import shlex
import gc
parent_folder_name = "data/"
folder_list=glob.glob(parent_folder_name+"*")
#for x in folder_list:
#    print(x)

#print(folder_list)
#
#c_f = 0
#c_g = 0
#
#out_file = open("hdd_failed_dataset.csv","a+")
#
#for files in folder_list:
#    back_slash_index = files.rfind("/")
#    file_name = files[back_slash_index+1:]
#    print(file_name)
#    
#    count = 0
#    with open(files,"r") as in_file:
#        for line in in_file:
##            print(line)
#            
#            parts = line.split(",")
##            print(parts[4])
#            
#            if(parts[4]!="failure"):
##                print(type(parts[4]))
#                if(int(parts[4])==1):
#                    out_file.write(line)
#                    out_file.flush()
#                    c_f+=1
#                if(int(parts[4])==0):
#                     c_g+=1     
#            
##            count+=1
##            if(count==10):
##                break
#
#print(c_f)
#print(c_g)                         
#print((c_f/(c_f+c_g))*100)

hdd_serial_number_list = []


with open("hdd_failed_dataset.csv","r") as in_file:
        for line in in_file:
#            print(line)
            parts = line.split(",")
#            print(parts)
            hdd_serial_number_list.append(parts[1])
print(len(hdd_serial_number_list))


#for hdd_serial_number in hdd_serial_number_list:
#    year = 2013 
#    print(hdd_serial_number)
#    while(year<=2019):
#        for month in range(1,13):
#            for day in range(1,32):
#                if month<=9:
#                        month_str = "0"+str(month)
#                else:
#                    month_str = str(month)
#                
#                if day<=9:
#                    day_str = "0"+str(day)
#                else:
#                    day_str = str(day)
#                
#                file_name = "data/"+str(year)+"-"+month_str+"-"+day_str+".csv"
#                try:
#                    with open(file_name,"r") as in_file:
#                        for line in in_file:
#                           if str(hdd_serial_number) in line:
#                               out_file = open("output_files/"+hdd_serial_number+".csv","a+")
#                               out_file.write(line)
#                               out_file.flush()
#                               break
#                except:
#                   print(file_name +" not found")
#                    
#                    
#        year+=1
            
                     