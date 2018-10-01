# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 23:49:19 2018

@author: Owner
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.datasets import make_moons
import seaborn as sns
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA
df_wine = pd.read_csv('wine.csv')

df_wine.shape
df_wine.info()
df_wine.head()
df_wine.describe()
cols = list(df_wine.columns)
#print(cols)
df_wine = df_wine.dropna()
for i in cols:
    plt.figure()
    sns.boxplot(x=i,data=df_wine)
sns.pairplot(df_wine[cols], size=2.5)
plt.tight_layout()
plt.show()
cm = np.corrcoef(df_wine[cols].values.T)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 6},
                 yticklabels=cols,
                 xticklabels=cols)
plt.tight_layout()
plt.show()
X,y = df_wine.iloc[:,:-1].values, df_wine['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
#print(X_train)
#print(X_train_std)
X_test_std = sc.transform(X_test)

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
plt.xlabel('feature')
plt.ylabel('predictor')
plt.savefig('3.png', dpi=300)
plt.show()


lr = LogisticRegression()
lr = lr.fit(X_train_std, y_train)
print('lr train R^2: %.2f' % lr.score(X_train_std, y_train))
print('lr test R^2: %.2f' % lr.score(X_test_std, y_test))


sv=svm.SVC()

sv = sv.fit(X_train_std, y_train)

print('sv train R^2: %.2f' % sv.score(X_train_std, y_train))
print('sv test R^2: %.2f' % sv.score(X_test_std, y_test))







def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[0],
                    label=cl)
        
pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

lr = lr.fit(X_train_pca, y_train)

plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
#plt.legend(loc='lower left')
plt.show()
print('pca lr train R^2: %.2f' % lr.score(X_train_pca, y_train))

plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
#plt.legend(loc='lower left')
plt.show()

X_train_pca = pca.fit_transform(X_train_std)
print(pca.explained_variance_ratio_)       
print('pca lr test R^2: %.2f' % lr.score(X_test_pca, y_test))

pca = PCA(n_components = 2)
sv=svm.SVC()
sv = sv.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=sv)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
#plt.legend(loc='lower left')
plt.show()
print('pca sv train R^2: %.2f' % sv.score(X_train_pca, y_train))

plot_decision_regions(X_test_pca, y_test, classifier=sv)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
#plt.legend(loc='lower left')
plt.show()
print('pca sv test R^2: %.2f' % sv.score(X_test_pca, y_test))





lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
print('lda lr trainR^2: %.2f' % lr.score(X_train_lda, y_train))
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
print('lda lr test R^2: %.2f' % lr.score(X_test_lda, y_test))
sv=svm.SVC()
sv = sv.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=sv)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
print('lda sv train R^2: %.2f' % sv.score(X_train_lda, y_train))
plot_decision_regions(X_test_lda, y_test, classifier=sv)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
print('lda sv test R^2: %.2f' % sv.score(X_test_lda, y_test))





gamma_space = np.logspace(-3,0,3)
for i in gamma_space:
    kpca = KernelPCA(n_components=2,kernel='rbf',gamma=i)
    X_train_kpca = kpca.fit_transform(X_train_std, y_train)
    X_test_kpca = kpca.transform(X_test_std)
    lr = lr.fit(X_train_kpca, y_train)
    plot_decision_regions(X_train_kpca, y_train, classifier=lr)
    plt.xlabel('KPCA 1')
    plt.ylabel('KPCA 2')
    plt.legend(loc='lower left')
    print('gamma '+str(i))
    plt.show()
    print('kpca lr trainR^2: %.2f' % lr.score(X_train_kpca, y_train))
    plot_decision_regions(X_test_kpca, y_test, classifier=lr)
    plt.xlabel('KPC 1')
    plt.ylabel('KPC 2')
    plt.legend(loc='lower left')
    print('gamma '+str(i))
    plt.show()
    print('kpca lr test R^2: %.2f' % lr.score(X_test_kpca, y_test))
    sv=svm.LinearSVC()
    sv = sv.fit(X_train_kpca, y_train)
    plot_decision_regions(X_train_kpca, y_train, classifier=sv)
    plt.xlabel('KPC 1')
    plt.ylabel('KPC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    print('gamma '+str(i))
    plt.show()
    print('kpca sv train R^2: %.2f' % sv.score(X_train_kpca, y_train))   
    plot_decision_regions(X_test_kpca, y_test, classifier=sv)
    plt.xlabel('KPC 1')
    plt.ylabel('KPC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    print('gamma '+str(i))
    plt.show()
    print('kpca sv test R^2: %.2f' % sv.score(X_test_kpca, y_test))

print("My name is Biao Feng")
print("My NetID is: biaof2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

