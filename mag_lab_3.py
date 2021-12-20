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

def cov(x,y):
    mult.clear()
    for i in range(len(x)):
        mult.append(x[i]*y[i])
    return mean(mult)-mean(x)*mean(y)

x = list()
y = list()
z = list()

x1 = list()
y1 = list()
z1 = list()
f1 = list()
d1 = list()
e1 = list()

mult=list()
n=10000
mu=0.5
sigma1=0.15
sigma2=0.35
sigma3=0.2

def mean(x):
    return sum(x)/len(x)

def cov(x,y):
    mult.clear()
    for i in range(len(x)):
        mult.append(x[i]*y[i])
    return mean(mult)-mean(x)*mean(y)

for j in range(n):
    x=[]
    for i in range(n):
        x.append(random.gauss(mu,sigma1))

for j in range(n):
    y=[]
    for i in range(n):
        y.append(random.gauss(mu,sigma2))

alpha=(sigma2*sigma2)/(sigma1*sigma1 + sigma2*sigma2)

for j in range(n):
    z=[]
    for i in range(n):
        z.append(alpha*x[i]+(1-alpha)*y[i])

print("A) New sigma: ", math.sqrt(cov(z,z)))



for j in range(n):
    x1=[]
    for i in range(n):
        x1.append(random.gauss(mu,sigma1))

for j in range(n):
    y1=[]
    for i in range(n):
        y1.append(random.gauss(mu,sigma2))

for j in range(n):
    z1=[]
    for i in range(n):
        z1.append(random.gauss(mu,sigma3))

alpha1=(sigma2*sigma3)/(sigma1*sigma2+sigma2*sigma3+sigma1*sigma3)
alpha2=(sigma1*sigma3)/(sigma1*sigma2+sigma2*sigma3+sigma1*sigma3)

sig=(1/(sigma1*sigma1))+(1/(sigma2*sigma2))+(1/(sigma3*sigma3))

for j in range(n):
    f1=[]
    for i in range(n):
        f1.append(alpha1*x1[i]+alpha2*y1[i]+(1-alpha1-alpha2)*z1[i])

print("B) New sigma №1, calculated by alpla1, alpha2, alpha3 formulas: ", math.sqrt(cov(f1,f1)))

for j in range(n):
    d1=[]
    for i in range(n):
        d1.append(((x1[i]/(sigma1*sigma1))+(y1[i]/(sigma2*sigma2))+(z1[i]/(sigma3*sigma3)))/(sig))

print("B) New sigma №2, calculated by 1/(sigma^2) formula: ", math.sqrt(cov(d1,d1)))

alpha3=(cov(z,z))/(sigma3*sigma3+cov(z,z))

for j in range(n):
    e1=[]
    for i in range(n):
        e1.append(alpha3*z1[i]+(1-alpha3)*z[i])

print("B) New sigma №3, calculated with the sigma from A) and a new sigma3=0.2. This method is similar to A) item: ", math.sqrt(cov(e1,e1)))

plt.subplot(4, 4, 1)
plt.title("sigma1=0.15")
plt.hist(x, range=[-1, 2], bins = 150, color = 'b')
plt.subplot(4, 4, 2)
plt.title("sigma2=0.35")
plt.hist(y, range=[-1, 2], bins = 150, color = 'r')
plt.subplot(4, 4, 3)
plt.title("Optimum sigma")
plt.hist(z, range=[-1, 2], bins = 150, color = 'g')
plt.subplot(4, 4, 4)
plt.title("sigma1=0.15, sigma2=0.35, Optimum sigma")
plt.hist(y, range=[-1, 2], bins = 150, color = 'b')
plt.hist(x, range=[-1, 2], bins = 150, color = 'r')
plt.hist(z, range=[-1, 2], bins = 150, color = 'g')
plt.subplot(4, 4, 5)
plt.title("sigma1=0.15")
plt.hist(x1, range=[-1, 2], bins = 150, color = 'm')
plt.subplot(4, 4, 6)
plt.title("sigma2=0.35")
plt.hist(y1, range=[-1, 2], bins = 150, color = 'r')
plt.subplot(4 ,4, 7)
plt.title("sigma3=0.20")
plt.hist(z1, range=[-1, 2], bins = 150, color = 'b')
plt.subplot(4, 4, 8)
plt.title("Optimum sigma, calculated by alpla1, alpha2, alpha3 formulas")
plt.hist(f1, range=[-1, 2], bins = 150, color = 'g')
plt.subplot(4, 4, 9)
plt.title("sigma1, sigma2, sigma3, Optimum sigma")
plt.hist(y1, range=[-1, 2], bins = 150, color = 'm')
plt.hist(z1, range=[-1, 2], bins = 150, color = 'r')
plt.hist(x1, range=[-1, 2], bins = 150, color = 'b')
plt.hist(f1, range=[-1, 2], bins = 150, color = 'g')
plt.subplot(4, 4, 10)
plt.title("Optimum sigma, calculated by 1/(sigma^2) formula")
plt.hist(d1, range=[-1, 2], bins = 150, color = 'g')
plt.subplot(4, 4, 11)
plt.title("sigma1, sigma2, sigma3, Optimum sigma another method")
plt.hist(y1, range=[-1, 2], bins = 150, color = 'm')
plt.hist(z1, range=[-1, 2], bins = 150, color = 'r')
plt.hist(x1, range=[-1, 2], bins = 150, color = 'b')
plt.hist(d1, range=[-1, 2], bins = 150, color = 'g')
plt.subplot(4, 4, 12)
plt.title("Optimum sigma, calculated by the sigma from A) with a new sigma of 0.2")
plt.hist(e1, range=[-1, 2], bins = 150, color = 'g')
plt.subplot(4, 4, 13)
plt.title("sigma1, sigma2, sigma3, Optimum sigma and another method")
plt.hist(y1, range=[-1, 2], bins = 150, color = 'm')
plt.hist(z1, range=[-1, 2], bins = 150, color = 'r')
plt.hist(x1, range=[-1, 2], bins = 150, color = 'b')
plt.hist(e1, range=[-1, 2], bins = 150, color = 'g')
plt.show()
