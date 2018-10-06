# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 09:04:22 2018

@author: Owner
"""

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import pandas as pd
Iris = datasets.load_iris()
columns = Iris.feature_names
y = Iris.target
X = Iris.data

lst_train = []
lst_test =[]
tree = DecisionTreeClassifier(max_depth= 4, min_samples_leaf=0.26)
columns1 = []
#columns2 = []
for i in range(1,11):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1, random_state=i,stratify=y)
    tree.fit(X_train, y_train)
    y_pred_test = tree.predict(X_test)
    train_score = tree.score(X_train,y_train)
    lst_train.append(train_score)
    test_score = tree.score(X_test, y_test)
    lst_test.append(test_score)
    columns1.append('score_'+str(i))
#    columns2.append('score_'+str(i))
train_mean = np.mean(lst_train)
train_std = np.std(lst_train)
test_mean = np.mean(lst_test)
test_std = np.std(lst_test)
df = pd.DataFrame([lst_train+[train_mean, train_std],lst_test+[test_mean, test_std]], index = ['train','test'],columns = columns1+['mean','std'])
#df_test = pd.DataFrame([lst_test+[test_mean, test_std]],  columns = columns2+['test_mean','test_std'])
print(df)


tree = DecisionTreeClassifier(max_depth= 4, min_samples_leaf=0.26)
columns1[10:] = ['cv_mean','cv_test']
columns1.append('out_of_sample')
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1, random_state=42,stratify=y)
cv_score = cross_val_score(tree, X_train, y_train, cv=10, scoring = 'accuracy')
cv_mean = np.mean(cv_score)
cv_std = np.std(cv_score)
tree.fit(X_train,y_train)
test_score = tree.score(X_test, y_test)
cv_score = list(cv_score)+[cv_mean, cv_std, test_score]

df = pd.DataFrame([cv_score],columns = columns1)
print(df)

print("My name is Biao ")
print("My NetID is: Feng")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

