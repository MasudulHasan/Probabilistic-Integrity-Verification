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


parent_folder = "/content/drive/My Drive/production/PrepareDataset/"
# folder_list=glob.glob(parent_folder+"*")
# print(folder_list)

mp={}
count_mp={}
for node in range(1,4):
  folder = parent_folder+"/dm"+str(node)+"_day_waise/"
  folder_list= glob.glob(folder+"*")
  # print(len(folder_list))

  count =0 

  for file_ in folder_list:
    # print(file_)
    print(node, count)
    count+=1
    start_index = file_.rfind(".log")
    end_index = file_.rfind(".txt")
    date = file_[start_index+5:end_index]

    with open(file_,"r+") as in_file:
      t_file_size = 0
      t_count =0
      for line in in_file:
        # print(line)
        try:
          s_index = line.rfind("NBYTES")
          e_index = line.rfind("VOLUME")

          file_size = line[s_index+7:e_index].strip()
          t_file_size+=int(file_size)
          t_count+=1
          # print(file_size)
        except:
          traceback.print_exc()
      
      if date not in mp:
        mp[date] = t_file_size
      else:
        v = mp[date]
        mp[date] = t_file_size+v
      
      if date not in count_mp:
        count_mp[date] = t_count
      else:
        v = count_mp[date]
        count_mp[date] = t_count+v


        
        # break


file_size_file = open("/content/drive/My Drive/file_size.txt","a+")
for k in mp:
  file_size_file.write(str(k)+" "+str(mp[k])+"\n")
  file_size_file.flush()

file_size_file.close()


count_file = open("/content/drive/My Drive/workload.txt","a+")
for k in count_mp:
  count_file.write(str(k)+" "+str(count_mp[k])+"\n")
  count_file.flush()

count_file.close()