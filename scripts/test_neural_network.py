#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 05:31:15 2019

@author: masudulhasanmasudb
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn import metrics
from imblearn.over_sampling import (RandomOverSampler, SMOTE, ADASYN)
from collections import Counter

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    
    print(TP/(TP+FN)) 
    
    print(FP/(FP+TN))       
           
    return(TP, FP, TN, FN)


hdd = pd.read_csv('../dataset_1.csv') 
print(hdd.shape)
print(hdd.describe().transpose())
#hdd = hdd.dropna()
#print(hdd.iloc[0]) 
hdd = hdd.drop(hdd.columns[6], axis=1)
hdd = hdd.drop(hdd.columns[9], axis=1)
#print(hdd.iloc[0])

hdd = hdd.dropna()

hdd_pos = hdd.loc[hdd[hdd.columns[12]] == 1]
hdd = hdd[hdd[hdd.columns[12]] == 0]
print(len(hdd))
hdd_neg = hdd
hdd_merged = [hdd_pos, hdd_neg]
result = pd.concat(hdd_merged)
x = result.iloc[:, :-1].values
y = result.iloc[:, -1].values

X_resampled, y_resampled = SMOTE().fit_resample(x, y)
print(sorted(Counter(y_resampled).items()))
print(len(y_resampled))


#max_th = dataset.iloc[:, 13:14].values

#print(x[:10])
#print(y[:10])
#print(len(x))
#print(type(x))




#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
#
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(100,50,50),activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

predict_prob = mlp.predict_proba(X_test)
print(predict_prob)

from sklearn.metrics import classification_report,confusion_matrix
#print(confusion_matrix(y_train,predict_train))
#print(classification_report(y_train,predict_train))
#
#print(confusion_matrix(y_test,predict_test))
#print(classification_report(y_test,predict_test))

print("Accuracy: ",metrics.accuracy_score(y_test, predict_test))
print("Recall: ",metrics.recall_score(y_test, predict_test))
conf_matrix = metrics.confusion_matrix(y_test, predict_test)
print(conf_matrix)
print(perf_measure(y_test, predict_test)) 
#


#print(count_unique(y_train))
#print(count_unique(y_test))
##
##real_x = dataset.iloc[:, :].values
#
##from sklearn.ensemble import RandomForestClassifier
##
###Create a Gaussian Classifier
##clf=RandomForestClassifier(n_estimators=500)
#clf=RandomForestClassifier()
##from sklearn import tree
##clf = tree.DecisionTreeClassifier()
#
##from sklearn.naive_bayes import GaussianNB
##clf = GaussianNB()
#
##from sklearn.svm import SVC
##clf = SVC(gamma='auto')
#
##regr = RandomForestRegressor(max_depth=2, random_state=0,
##                              n_estimators=100)
#
##X_train = X_train.fillna(X_train.mean())
#
##print(np.where(np.isnan(X_train)))
##print(type(X_train))
##print(np.where(np.isnan(y_train)))
#
##Train the model using the training sets y_pred=clf.predict(X_test)
#clf.fit(X_train,y_train)
#y_pred=clf.predict(X_test)
#preds = clf.predict_proba(X_test)
#
##preds = clf.predict_proba(X_test)
##out_file = open("output.txt","a+")
##for x in preds:
##    out_file.write(str(x)+"\n")
##    out_file.flush()
##print(preds[:,1])
##print(collections.Counter(preds[:,1]))
##print(preds.shape)
##print(preds)
##print(preds[:,1])
##print(test_labels)
#
#print(y_pred)
#print(preds)
#
#for i in range(len(y_pred)): 
#    if y_test[i]!=y_pred[i]:
#        print(str(y_test[i])+" "+str(y_pred[i])+" "+str(preds[i]))
#           
#
#print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
#print("Recall: ",metrics.recall_score(y_test, y_pred))
#conf_matrix = metrics.confusion_matrix(y_test, y_pred)
#print(conf_matrix)
#print(perf_measure(y_test, y_pred)) 
#
#import seaborn as sn
#import pandas as pd
#import matplotlib.pyplot as plt
#
#df_cm = pd.DataFrame(conf_matrix, range(2),
#                  range(2))
#plt.figure(figsize = (8,6))
#sn.set(font_scale=1.4)#for label size
#sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
#
#plt.show()             