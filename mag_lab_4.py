import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import os
import sys
import time
import datetime
import re
import statistics
import random
import math
from statistics import mean
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import f
from scipy import stats

def mean(x):
    return sum(x)/len(x)

def mean_ch(x,n_count):
    return sum(np.multiply(x,n_count))/(sum(n_count))

def sig_ch(x,n_count):
    mu=mean_ch(x,n_count)
    return sum(n_count*(np.array(x) - mu)*(np.array(x) - mu))/(sum(n_count)-1)

def cov(x,y):
    mult.clear()
    for i in range(len(x)):
        mult.append(x[i]*y[i])
    return mean(mult)-mean(x)*mean(y)

def gaussian(x, mu, sig):
    return np.exp(-np.power((x - mu)/sig, 2.)/2)

data_50=list()
data_500=list()
i=0

file=open("C:/Users/HYPERPC/Desktop/chi2task_v5.dat", "r")

#for line in file:
#    i+=1
#    if (i>50):
#        data_50.append(float(line.split()[0]))
#        if (i==100):
#            break

for line in file:
    i+=1
    if (i>=50):
        data_50.append(float(line.split()[0]))
        if (i==100):
            break

for line in file:
    i+=1
    data_500.append(float(line.split()[0]))


#s=sum(data_50)
#mu=mean(s)
#sigma=np.sqrt(cov(s,s))
#s_1=(sum(data_50)-mu)/(sigma)
#heights1,edges1,patches1 = plt.hist(s, bins=30)
#heights2,edges2,patches2 = plt.hist(s_1, bins=30)
##heights1,edges1,patches1 = plt.hist(data_50, bins=60)
##x_values1 = np.linspace(edges1[0], edges1[-1], 1000)
#x_values2 = np.linspace(edges2[0], edges2[-1], 1000)
##xA_ch1= (edges1[1:] + edges1[:-1])/2
##widths1 = edges1[1:] - edges1[:-1]
##sigma1=np.sqrt(sig_ch(xA_ch1,heights1))
##totalWeight1 = (heights1*widths1).sum()/(np.sqrt(2.*np.pi)*sigma1)

##mu=mean_ch(xA_ch1,heights1)
##sigma=np.sqrt(sig_ch(xA_ch1,heights1))

bins1=20
bins2=15
bins3=20

heights1,edges1,patches1 = plt.hist(data_50, bins=bins1 ,color="b");
xA_ch1= (edges1[1:] + edges1[:-1])/2
mu1=mean_ch(xA_ch1,heights1)
sigma1=np.sqrt(sig_ch(xA_ch1,heights1))
x_values1 = np.linspace(edges1[0], edges1[-1], 1000)
widths1 = edges1[1:] - edges1[:-1]
totalWeight1 = (heights1*widths1).sum()/(np.sqrt(2.*np.pi)*sigma1)

heights2,edges2,patches2 = plt.hist(data_500, bins=bins2 ,color="b");
xA_ch2= (edges2[1:] + edges2[:-1])/2
mu2=mean_ch(xA_ch2,heights2)
sigma2=np.sqrt(sig_ch(xA_ch2,heights2))
x_values2 = np.linspace(edges2[0], edges2[-1], 1000)
widths2 = edges2[1:] - edges2[:-1]
totalWeight2 = (heights2*widths2).sum()/(np.sqrt(2.*np.pi)*sigma2)



def Chauv_cr(x):
    x=np.array(x)
    print("At first:\n",x)
    while True:
        mu1=sum(x)/len(x)
        sigma1=np.sqrt(sum((x - mu1)*(x - mu1))/(len(x)-1))
        z=max(abs(mu1-x)/sigma1)
        z_i=(abs(mu1-x)/sigma1).argmax()

        perc=int(100000) #parameter of precision for integral (if bin lenth bigger - than presision lower)

        #calculating likelihood by integral with descritisation n=10^5
        Phi=(gaussian(np.linspace(0, z, perc), 0, 1)*z/(np.sqrt(2.*np.pi)*perc)).sum()
        if (1-2*Phi)*len(x)<0.5:
            x = np.delete(x, z_i)
        else:
            break
    return x

y=Chauv_cr(data_50)
print("After:",y)

heights3,edges3,patches3 = plt.hist(y, bins=bins3 ,color="b");
xA_ch3= (edges3[1:] + edges3[:-1])/2
mu3=mean_ch(xA_ch3,heights3)
sigma3=np.sqrt(sig_ch(xA_ch3,heights3))
x_values3 = np.linspace(edges3[0], edges3[-1], 1000)
widths3 = edges3[1:] - edges3[:-1]
totalWeight3 = (heights3*widths3).sum()/(np.sqrt(2.*np.pi)*sigma3)

c1=widths1/(np.sqrt(2.*np.pi)*sigma1*totalWeight1)
n1=1/c1
perc1=int(10000/bins1*22*2**((bins1-22)/300))
ei1=[(gaussian(np.linspace(edges1[i], edges1[i+1], perc1), mu1, sigma1)*widths1[i]/(np.sqrt(2.*np.pi)*sigma1*perc1)).sum() for i in range(len(widths1))]
xi1=((np.array(heights1)*c1-np.array(ei1))*(np.array(heights1)*c1-np.array(ei1))/(np.array(ei1)*c1))
print("\nChi^2 parametr for 50 values =",xi1.sum(), "\n")
print("Degree of freedom for 50 values = ",len(xi1)-3, "\n")

c2=widths2/(np.sqrt(2.*np.pi)*sigma2*totalWeight2)
n2=1/c2
perc2=int(10000/bins2*22*2**((bins2-22)/300))
ei2=[(gaussian(np.linspace(edges2[i], edges2[i+1], perc2), mu2, sigma2)*widths2[i]/(np.sqrt(2.*np.pi)*sigma2*perc2)).sum() for i in range(len(widths2))]
xi2=((np.array(heights2)*c2-np.array(ei2))*(np.array(heights2)*c2-np.array(ei2))/(np.array(ei2)*c2))
print("\nChi^2 parametr for 500 values =",xi2.sum(), "\n")
print("Degree of freedom for 500 values = ",len(xi2)-3, "\n")

c3=widths3/(np.sqrt(2.*np.pi)*sigma3*totalWeight3)
n3=1/c3
perc3=int(10000/bins3*22*2**((bins3-22)/300))
ei3=[(gaussian(np.linspace(edges3[i], edges3[i+1], perc3), mu3, sigma3)*widths3[i]/(np.sqrt(2.*np.pi)*sigma3*perc3)).sum() for i in range(len(widths3))]
xi3=((np.array(heights3)*c3-np.array(ei3))*(np.array(heights3)*c3-np.array(ei3))/(np.array(ei3)*c3))
print("\nChi^2 parametr for 50 values (corrected) =",xi3.sum(), "\n")
print("Degree of freedom for 50 values (corrected) = ",len(xi3)-3, "\n")

plt.subplot(2, 2, 1)
plt.title("Hystogram for 50 values")
plt.hist(data_50, range=[80, 105], bins = bins1, color = "b")
plt.plot(x_values1, totalWeight1*gaussian(x_values1, mu1, sigma1), color='tab:red')

plt.subplot(2, 2, 2)
plt.title("Hystogram for 500 values")
plt.hist(data_500, range=[70, 110], bins = bins2, color = "b")
plt.plot(x_values2, totalWeight2*gaussian(x_values2, mu2, sigma2), color='tab:red')

plt.subplot(2, 2, 3)
plt.title("Hystogram for 50 values (corrected)")
plt.hist(y, range=[80, 105], bins = bins3, color = "b")
plt.plot(x_values3, totalWeight3*gaussian(x_values3, mu3, sigma3), color='tab:red')

plt.show()
