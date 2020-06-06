#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 03:27:16 2019

@author: masudulhasanmasudb
"""
count=0
with open("final_log.txt","r")as in_file:
        for line in in_file:
            if(len(line.strip())>0):
#                parts = line.strip().split(" ")
#                if "self calculated" in line or "cheksem run on" in line :
#                    print(line)
#                
#                if "[" not in line and len(parts)==4:
#                print(line)
                count+=1
                print(count)
#                if "total_dik" in line:
#                    print(line)
#                if "check Sum run on" in line :
#                    print(line)
                if "perct" in line:
                    print(line)