#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 23:16:36 2019

@author: masudulhasanmasudb
"""
import time
import glob,random
import datetime
import os
import subprocess
import shlex
import gc
parent_folder_name = "../final_logs1"+"/"
folder_list=glob.glob(parent_folder_name+"*")
print(folder_list)

for i in range(1,8):
    for folder in folder_list:
        if "/"+str(i) in folder:
            file_list=glob.glob(folder+"/*")
        #    print(file_list)
            value_list =[]
            total_value_list = []
            model_name_list = []
            total_file_size_list =[]
            for file in file_list:
                if "old" not in file:
        #            print(file)
                    total_file = 0
                    total_file_size = 0
                    run_checksum_on = 0
                    can_count = False
                    with open(file,"r")as in_file:
                        for line in in_file:
                            if "total_disk" in line:
        #                        print(line)
                                parts = line.split(" ")
                                total_file+=int(parts[1])
                                can_count = True
                            if "check Sum run on" in line and can_count:
                                index = line.rfind("on")
                                value = int(line[index+3:].strip())
                                can_count = False
                                run_checksum_on+=value
        #                        if(value >= 3000):
        #                            print(file)
        #                        print(value)
                            
                            if "file size" in line:
                                parts = line.split(" ")
                                total_file_size+=int(parts[2])
                                    
                    print(total_file)
                    print(run_checksum_on)
    #                print(file)
                    f_index = file.rfind("/")
                    e_index = file.rfind(".txt")
                    model_name = file[f_index+1:e_index]
                    print(model_name)
                    print(run_checksum_on/total_file)
#                    value_list.append((run_checksum_on/total_file)*100)
                    value_list.append(run_checksum_on)
                    model_name_list.append(model_name)
                    total_value_list.append(total_file)
                    total_file_size_list.append(total_file_size)
            
            print(model_name_list)
            print(value_list)
            print(total_file_size_list)
            
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(12, 6), dpi=100)
            ax = fig.add_axes([0,0,1,1])
            width = 0.35
#            ax.bar(model_name_list,value_list, width)
            ax.bar(model_name_list, total_value_list, width, color='r')
            ax.bar(model_name_list, value_list, width,bottom=total_value_list, color='b')
#            plt.suptitle("For "+str(i)+" days",fontsize=20)
            plt.savefig(str(i)+".png")
            plt.show()
            
            fig = plt.figure(figsize=(12, 6), dpi=100)
            ax = fig.add_axes([0,0,1,1])
            width = 0.35
#            ax.bar(model_name_list,value_list, width)
            a_list=[v for v in range(4)]
            plt.plot(a_list, total_file_size_list)
#            ax.bar(model_name_list, value_list, width,bottom=total_value_list, color='b')
#            plt.suptitle("For "+str(i)+" days",fontsize=20)
#            plt.savefig(str(i)+".png")
            plt.show()