#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 18:06:53 2019

@author: masudulhasanmasudb
"""
#hdd = pd.read_csv('../dataset_for_training/dataset_ST8000DM002.csv')
parent_folder_name = "../dataset_for_training/"
folder_list=glob.glob(parent_folder_name+"*")
#print(folder_list)

for x in folder_list:
    has_problem = False
    index = x.rfind("/")
    file_name = x[index+1:]
    print(file_name)
#    if "ST8000DM002" in x:
    line_list=[]
    with open(x) as in_file:
        for line in in_file:
            parts = line.split(",")
#            print(len(parts))
            count = 0
            for y in parts:
                if len(y.strip())>0:
                    count+=1
            if count==1:
                has_problem = True
                print(line)
#                print(count)
#            else:
#                line_list.append(line)
#    print(has_problem)
#    if has_problem:
#        out_file = open("../dataset_for_training/new_"+file_name,"a+")
#        for y in line_list:
#            out_file.write(y)
#            out_file.write("\n")
#            out_file.flush()


#for x in folder_list:
#    if "new" in x:
#        print(x)
#        index = x.rfind("new_")
#        file_name = x[index+4:]
#        print(file_name)
#        out_file = open("../dataset_for_training/"+file_name,"a+")
#
#        with open(x) as in_file:
#            for line in in_file:
#                if len(line.strip())>0:
#                    out_file.write(line)
#                    out_file.flush()
#                    
                        