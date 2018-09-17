from pandas import DataFrame
import matplotlib.pyplot as plot
from random import uniform
from math import sqrt
import pylab
import scipy.stats as stats
import pandas as pd
import urllib
import sys
import numpy as np

target_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")
data = urllib.request.urlopen(target_url)
xList = []
labels = []
for line in data:
    row = line.strip().split(",".encode())
    xList.append(row)
sys.stdout.write("Number of Rows of Data = " + str(len(xList)) + '\n')
sys.stdout.write("Number of Columns of Data = " + str(len(xList[1]))+'\n')


nrow = len(xList)
ncol = len(xList[1])
type = [0]*3
colCounts = []
for col in range(ncol):
    for row in xList:
        try:
            a = float(row[col]) 
            if isinstance(a, float):
                type[0] += 1
        except ValueError:
                if len(row[col]) > 0:
                    type[1] += 1
                else:
                    type[2] += 1
colCounts.append(type)
type = [0]*3
sys.stdout.write("Col#" + '\t' + "Number" + '\t' + "Strings" + '\t ' + "Other\n")
iCol = 0
for types in colCounts:
    sys.stdout.write(str(iCol) + '\t\t' + str(types[0]) + '\t\t' + str(types[1]) + '\t\t' + str(types[2]) + "\n")
    iCol += 1

col = 3
colData = []
for row in xList:
    colData.append(float(row[col]))
stats.probplot(colData, dist="norm", plot=pylab)
pylab.show()
colArray = np.array(colData)
colMean = np.mean(colArray)
colsd = np.std(colArray)
sys.stdout.write("Mean = " + '\t' + str(colMean) + '\t\t' + "Standard Deviation = " + '\t ' + str(colsd) + "\n")
ntiles = 4
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
sys.stdout.write("\nBoundaries for 4 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")
ntiles = 10
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
sys.stdout.write("Boundaries for 10 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")
percentBdry = []

for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
sys.stdout.write("Boundaries for 10 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")
col = 60
colData = []
for row in xList:
    colData.append(row[col])
unique = set(colData)
sys.stdout.write("Unique Label Values \n")
print(unique)
catDict = dict(zip(list(unique),range(len(unique))))
catCount = [0]*2
for elt in colData:
    catCount[catDict[elt]] += 1
sys.stdout.write("\nCounts for Each Value of Categorical Label \n")
print(list(unique))
print(catCount)

rocksVMines = pd.read_csv(target_url,header=None, prefix="V")
print(rocksVMines.head())
print(rocksVMines.tail())
summary = rocksVMines.describe()
print(summary)

for i in range(208):
    if rocksVMines.iat[i,60] == "M":
            pcolor = "red"
    else:
            pcolor = "blue"
    dataRow = rocksVMines.iloc[i,0:60]
    dataRow.plot(color=pcolor)
plot.xlabel("Attribute Index")
plot.ylabel(("Attribute Values"))
plot.show()


dataRow2 = rocksVMines.iloc[1,0:60]
dataRow3 = rocksVMines.iloc[2,0:60]
plot.scatter(dataRow2, dataRow3)
plot.xlabel("2nd Attribute")
plot.ylabel(("3rd Attribute"))
plot.show()
dataRow21 = rocksVMines.iloc[20,0:60]
plot.scatter(dataRow2, dataRow21)
plot.xlabel("2nd Attribute")
plot.ylabel(("21st Attribute"))
plot.show()

target = []
for i in range(208):
    if rocksVMines.iat[i,60] == "M":
        target.append(1.0)
    else:
        target.append(0.0)
dataRow = rocksVMines.iloc[0:208,35]
plot.scatter(dataRow, target)
plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()
target = []
for i in range(208):
    if rocksVMines.iat[i,60] == "M":
        target.append(1.0 + uniform(-0.1, 0.1))
    else:
        target.append(0.0 + uniform(-0.1, 0.1))
dataRow = rocksVMines.iloc[0:208,35]
plot.scatter(dataRow, target, alpha=0.5, s=120)
plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()



dataRow2 = rocksVMines.iloc[1,0:60]
dataRow3 = rocksVMines.iloc[2,0:60]
dataRow21 = rocksVMines.iloc[20,0:60]
mean2 = 0.0; mean3 = 0.0; mean21 = 0.0
numElt = len(dataRow2)
for i in range(numElt):
    mean2 += dataRow2[i]/numElt
    mean3 += dataRow3[i]/numElt
    mean21 += dataRow21[i]/numElt
var2 = 0.0; var3 = 0.0; var21 = 0.0
for i in range(numElt):
    var2 += (dataRow2[i] - mean2) * (dataRow2[i] - mean2)/numElt
    var3 += (dataRow3[i] - mean3) * (dataRow3[i] - mean3)/numElt
    var21 += (dataRow21[i] - mean21) * (dataRow21[i] - mean21)/numElt
corr23 = 0.0; corr221 = 0.0
for i in range(numElt):
    corr23 += (dataRow2[i] - mean2) * \
              (dataRow3[i] - mean3) / (sqrt(var2*var3) * numElt)
    corr221 += (dataRow2[i] - mean2) * \
               (dataRow21[i] - mean21) / (sqrt(var2*var21) * numElt)        
sys.stdout.write("Correlation between attribute 2 and 3 \n")
print(corr23)
sys.stdout.write(" \n")
sys.stdout.write("Correlation between attribute 2 and 21 \n")
print(corr221)
sys.stdout.write(" \n")


corMat = DataFrame(rocksVMines.corr())
plot.pcolor(corMat)
plot.show()



dataCol2 = rocksVMines.iloc[:,1]
dataCol3 = rocksVMines.iloc[:,2]
dataCol21 = rocksVMines.iloc[:,20]
plot.scatter(dataCol2, dataCol3)
plot.xlabel("2nd Attribute")
plot.ylabel(("3rd Attribute"))
plot.show()
plot.scatter(dataCol2, dataCol21)
plot.xlabel("2nd Attribute")
plot.ylabel(("21st Attribute"))
plot.show()
plot.scatter(dataRow, target)
print("My name is Biao Feng")
print("My NetID is: biaof2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
