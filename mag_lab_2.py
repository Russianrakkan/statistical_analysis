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
from scipy import integrate
from scipy import stats
from scipy.stats import chisquare

x = list()
s = list()
s_1 = list()
n = 3000
mult=list()

g1=list()
g2=list()

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

for j in range(n):
    x=[]
    for i in range(n):
        x.append(np.random.poisson(3))
    s.append(sum(x))

mu=mean(s)
sigma=np.sqrt(cov(s,s))
print("mu = ", mu, "\n")
print("sigma = ", sigma, "\n")

for j in range(n):
    x=[]
    for i in range(n):
        x.append(np.random.poisson(3))
    s_1.append((sum(x)-mu)/(sigma))

mu_std=mean(s_1)
sigma_std=np.sqrt(cov(s_1,s_1))
print("mu_std = ", mu_std, "\n")
print("sigma_std = ", sigma_std, "\n")

#heights,edges,patches = plt.hist(x, bins=30)
heights1,edges1,patches1 = plt.hist(s, bins=30)
heights2,edges2,patches2 = plt.hist(s_1, bins=30)
#xA_ch= (edges[1:] + edges[:-1])/2
xA_ch1= (edges1[1:] + edges1[:-1])/2
xA_ch2= (edges2[1:] + edges2[:-1])/2

#mu=mean_ch(xA_ch,heights)
mu1=mean_ch(xA_ch1,heights1)
mu2=mean_ch(xA_ch2,heights2)
#sigma=np.sqrt(sig_ch(xA_ch,heights))
sigma1=np.sqrt(sig_ch(xA_ch1,heights1))
sigma2=np.sqrt(sig_ch(xA_ch2,heights2))
#x_values = np.linspace(edges[0], edges[-1], 1000)
x_values1 = np.linspace(edges1[0], edges1[-1], 1000)
x_values2 = np.linspace(edges2[0], edges2[-1], 1000)
#widths = edges[1:] - edges[:-1]
widths1 = edges1[1:] - edges1[:-1]
widths2 = edges2[1:] - edges2[:-1]
#totalWeight = (heights*widths).sum()/(np.sqrt(2.*np.pi)*sigma)
totalWeight1 = (heights1*widths1).sum()/(np.sqrt(2.*np.pi)*sigma1)
totalWeight2 = (heights2*widths2).sum()/(np.sqrt(2.*np.pi)*sigma2)


for j in range(n):
    g1=[]
    for i in range(n):
        g1.append(random.gauss(mu,sigma))

for j in range(n):
    g2=[]
    for i in range(n):
        g1.append(random.gauss(mu_std,sigma_std))

xi1=list()
perc1=int(5000/30*22*2**((30-22)/300))
ei1=[(gaussian(np.linspace(edges1[i], edges1[i+1], perc1), mu1, sigma1)*widths1[i]/(np.sqrt(2.*np.pi)*sigma1*perc1)).sum() for i in range(len(widths1))]
c1=widths1/(np.sqrt(2.*np.pi)*sigma1*totalWeight1)
n1=1/c1
xi1=((heights1-ei1*n1)*(heights1-ei1*n1)/(ei1*n1))
print("Chi^2 parametr =",xi1.sum(), "\n")
print("Degree of freedom = ",len(xi1)-3, "\n")

xi2=list()
perc2=int(5000/30*22*2**((30-22)/300))
ei2=[(gaussian(np.linspace(edges2[i], edges2[i+1], perc2), mu2, sigma2)*widths2[i]/(np.sqrt(2.*np.pi)*sigma2*perc2)).sum() for i in range(len(widths2))]
c2=widths2/(np.sqrt(2.*np.pi)*sigma2*totalWeight2)
n2=1/c2
xi2=((heights2-ei2*n2)*(heights2-ei2*n2)/(ei2*n2))
print("Chi^2 parametr for std gaussian =",xi2.sum(), "\n")
print("Degree of freedom =",len(xi2)-3, "\n")
#print("Chi1= ", , "\n")
#print("Chi2= ", )


#print(x)
#print(s)
plt.subplot(221)
plt.hist(x, range=[0, 30], bins = 30)
#plt.plot(x_values, totalWeight*gaussian(x_values, mu, sigma), color='tab:red')
plt.subplot(222)
plt.hist(s, bins = 30)
plt.plot(x_values1, totalWeight1*gaussian(x_values1, mu, sigma), color='tab:red')
plt.subplot(223)
plt.hist(s_1, bins = 30)
plt.plot(x_values2, totalWeight2*gaussian(x_values2, mu_std, sigma_std), color='tab:red')
plt.show()
