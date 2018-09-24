# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
import scipy as sp
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
os.getcwd()
df = pd.read_excel('housing.xlsx')
df.shape
df.info()
df.head()
df.describe()
cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = df.dropna()
for i in cols:
    plt.figure()
    sns.boxplot(x=i,data=df)
    plt.savefig(i+'.png', dpi=300)
sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.savefig('1.png', dpi=300)
plt.show()
cm = np.corrcoef(df[cols].values.T)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 8},yticklabels=cols,xticklabels=cols)
plt.tight_layout()
plt.savefig('2.png', dpi=300)
plt.show()

X = df[['RM']].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=20)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return 
slr = LinearRegression()
slr.fit(X_train, y_train)
y_pred = slr.predict(X_train)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)
print('R^2: %.3f' % slr.score(X_train, y_train))
lin_regplot(X_train, y_train, slr)
plt.xlabel('RM ')
plt.ylabel('MEDV')
plt.savefig('3.png', dpi=300)
plt.show()
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)
np.set_printoptions(precision=3)
print('Slope:' , slr.coef_)
print('Intercept: %.3f' % slr.intercept_)
print('R^2: %.3f' % slr.score(X_train, y_train))
ary = np.array(range(100000))
plt.scatter(y_train_pred,  y_train_pred - y_train,c='steelblue', marker='o', edgecolor='white',label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,c='limegreen', marker='s', edgecolor='white',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper right')
plt.hlines(y=0, xmin=-10, xmax=90, color='black', lw=2)
plt.xlim([-10, 90])
plt.tight_layout()
plt.savefig('4.png', dpi=300)
plt.show()
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))
print('ridge regression')
alpha_space = np.logspace(-3, 0, 3)
ridge_scores = []
ridge_scores_std = []
ridge = Ridge(normalize=True)
for alpha in alpha_space:
    ridge.alpha = alpha
    ridge.fit(X_train, y_train)
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)
    print('Slope:' , ridge.coef_)
    print('Intercept: %.3f' % ridge.intercept_)
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))
    ary = np.array(range(100000))
    plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=90, color='black', lw=2)
    plt.xlim([-10, 90])
    plt.tight_layout()
    plt.title('alpha='+str(alpha))
    plt.savefig('ridge alpha='+str(alpha)+'4.png', dpi=300)
    plt.show()
alpha_space = np.logspace(-3, 0, 3)
ridge_scores = []
ridge_scores_std = []
ridge = Ridge(normalize=True)
for alpha in alpha_space:
    ridge.alpha = alpha
    ridge_cv_scores = cross_val_score(ridge, X_train, y_train, cv=10)
    ridge_scores.append(np.mean(ridge_cv_scores))
    ridge_scores_std.append(np.std(ridge_cv_scores))
display_plot(ridge_scores, ridge_scores_std,'ridge')
print('lasso regression')
alpha_space = np.logspace(-3, 0, 3)
lasso_scores = []
lasso_scores_std = []
lasso = Lasso(normalize=True)
for alpha in alpha_space:
    lasso.alpha = alpha
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
    print('Slope:' , lasso.coef_)
    print('Intercept: %.3f' % lasso.intercept_)
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))
    ary = np.array(range(100000))
    plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=90, color='black', lw=2)
    plt.xlim([-10, 90])
    plt.tight_layout()
    plt.title('alpha='+str(alpha))
    plt.savefig('lasso alpha='+str(alpha)+'4.png', dpi=300)
    plt.show()
alpha_space = np.logspace(-3, 0, 3)
lasso_scores = []
lasso_scores_std = []
lasso = Lasso(normalize=True)
for alpha in alpha_space:
    lasso.alpha = alpha
    lasso_cv_scores = cross_val_score(lasso, X_train, y_train, cv=10)
    lasso_scores.append(np.mean(lasso_cv_scores))
    lasso_scores_std.append(np.std(lasso_cv_scores))
display_plot(lasso_scores, lasso_scores_std,'lasso')
print('Elastic Net regression')
alpha_space = np.logspace(-3, 0, 3)
elanet_scores = []
elanet_scores_std = []
elanet = ElasticNet(alpha=1.0)
for alpha in alpha_space:
    elanet.l1_ratio = alpha
    elanet.fit(X_train, y_train)
    y_train_pred = elanet.predict(X_train)
    y_test_pred = elanet.predict(X_test)
    print('Slope:' , elanet.coef_)
    print('Intercept: %.3f' % elanet.intercept_)
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))
    ary = np.array(range(100000))
    plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper right')
    plt.hlines(y=0, xmin=-10, xmax=90, color='black', lw=2)
    plt.xlim([-10, 90])
    plt.tight_layout()
    plt.title('l1_ratio='+str(alpha))
    plt.savefig('ElasticNet l1_ratio='+str(alpha)+'4.png', dpi=300)
    plt.show()
alpha_space = np.logspace(-3, 0, 3)
elanet_scores = []
elanet_scores_std = []
elanet = ElasticNet(alpha=1.0,l1_ratio=0.5,normalize=True)
for alpha in alpha_space:
    elanet.alpha = alpha
    elanet_cv_scores = cross_val_score(elanet, X_train, y_train, cv=10)
    elanet_scores.append(np.mean(elanet_cv_scores))
    elanet_scores_std.append(np.std(elanet_cv_scores))
display_plot(elanet_scores, elanet_scores_std,'elanet')
print("My name is Biao Feng")
print("My NetID is: biaof2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
