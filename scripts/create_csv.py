#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:33:52 2020

@author: masudulhasanmasudb
"""

#out_file = open("data.csv","a+")
#file_size_list=[]
#workload_list=[]
#with open("/home/masudulhasanmasudb/eclipse-workspace/Proactive_error_detection/file_size.txt","r+") as in_file:
#    for line in in_file:
#        s_index = line.rfind(":")
#        value = int(line[s_index+1:].strip())
##        print(value)
#        file_size_list.append(value)
#        
#with open("/home/masudulhasanmasudb/eclipse-workspace/Proactive_error_detection/workload.txt","r+") as in_file:
#    for line in in_file:
#        s_index = line.rfind(":")
#        value = int(line[s_index+1:].strip())
##        print(value)
#        workload_list.append(value)        
#
#print(len(file_size_list))
#print(len(workload_list))
#
#d = {
#      "wl": workload_list
#}

count= 0
with open("/home/masudulhasanmasudb/eclipse-workspace/Proactive_error_detection/map_2018.txt","r")as in_file:
    for line in in_file:
        parts = line.split(" ")
#        print(parts)
        if len(parts)==2:
            if "1" in parts[1].strip():
                count+=1
                print(line)
print(count)