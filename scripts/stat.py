#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:13:41 2019

@author: masudulhasanmasudb
"""
count =0
number_of_occurance =0
TP=0
FN = 0
FP = 0
TN = 0
with open("final_log.txt","r")as in_file:
        for line in in_file:
            if(len(line.strip())>0):
                if("1," in line):
#                    print(line)
#                    print(str(line).count("1,"))
                    number_of_occurance+=int(str(line).count("1,"))
                    count+=1
                
                if "[" not in line and "next" not in line:
                    parts = line.strip().split(" ")    
                    if(len(parts)==4):
                        TP += int(parts[0])
                        FN += int(parts[3])
                        FP += int(parts[1])
                        TN += int(parts[2])
#                        print(line)
#                    count+=1    

print(count)         
print(number_of_occurance) 
print(TP)
print(FN)
print(FP)
print(TN)

print(FP/(FP+TN))      
print(TP/(TP+FN)) 

print(TP/(TP+FP)) 

print(TP+FP+TN+FN)     