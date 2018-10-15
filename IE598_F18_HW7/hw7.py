# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 23:54:46 2018

@author: Owner
"""
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import numpy as np
file = pd.read_csv('wine.csv')
X,y = file.iloc[:,:-1].values, file['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=42)
lst_test = []
lst_train = []
est = [i for i in range(1,100)]
for i in est:
    rf = RandomForestClassifier(n_estimators = i)
    rf.fit(X_train, y_train)
    lst_train.append(rf.score(X_train,y_train))
    lst_test.append(rf.score(X_test,y_test))
frame = pd.DataFrame([est, lst_train, lst_test], index = ['estimators', 'insample_score','outsample_score'], columns =None)
print(frame)




params_rf = {
    'n_estimators':est
}
rf = RandomForestClassifier(random_state = 42)
grid_rf = GridSearchCV(estimator= rf,
                       param_grid= params_rf,
                       scoring="accuracy",
                       cv=10,
                       
                       )
#print(len(X_train),len(y_train))
#print(type_of_target(X_train))


grid_rf.fit(X_train,y_train)
best_model = grid_rf.best_estimator_
print(grid_rf.best_params_)
#print(best_model.score(X_test,y_test))
feat_labels = file.columns[:-1]
#print(X_train.shape[1])
#forest = RandomForestClassifier(n_estimators=500,random_state=1)
#forest.fit(X_train, y_train)
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
print(indices)

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),importances[indices],align='center')
plt.xticks(range(X_train.shape[1]),feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
