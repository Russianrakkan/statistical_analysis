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

test_x_1_list=list()
test_y_1_list=list()

test_x_2_list=list()
test_y_2_list=list()

test_x_3_list=list()
test_y_3_list=list()
test_z_3_list=list()

n=10000
for i in range(n):
    test_x_1=random.random()
    test_y_1=random.random()
    test_x_1_list.append(test_x_1)
    test_y_1_list.append(test_y_1)

    test_x_2=random.random()
    test_y_2=test_x_2*(-5)+0.6
    test_x_2_list.append(test_x_2)
    test_y_2_list.append(test_y_2)

    test_x_3=random.random()
    test_z_3=random.random()
    test_y_3=test_x_3*test_z_3
    test_x_3_list.append(test_x_3)
    test_z_3_list.append(test_z_3)
    test_y_3_list.append(test_y_3)

matplotlib.rc('xtick', labelsize=25)
matplotlib.rc('ytick', labelsize=25)

xy_1=list()
x_sqr_1=list()
y_1=list()

xy_2=list()
x_sqr_2=list()
y_2=list()

xy_3=list()
x_sqr_3=list()
y_3=list()

#chto_x=[3,4,2]
#chto_y=[15,5,1]

n_1=len(test_x_1_list)

n_2=len(test_x_2_list)

n_3=len(test_x_3_list)

for i in range(len(test_x_1_list)):
    xy_1.append(test_x_1_list[i]*test_y_1_list[i])
    x_sqr_1.append(test_x_1_list[i]*test_x_1_list[i])
a_1=((n_1*sum(xy_1)-sum(test_x_1_list)*sum(test_y_1_list))/(n_1*sum(x_sqr_1)-(sum(test_x_1_list)*sum(test_x_1_list))))
b_1=(sum(test_y_1_list)-a_1*sum(test_x_1_list))/n_1
for i in range(len(test_x_1_list)):
    y_1.append(a_1*test_x_1_list[i]+b_1)

for i in range(len(test_x_2_list)):
    xy_2.append(test_x_2_list[i]*test_y_2_list[i])
    x_sqr_2.append(test_x_2_list[i]*test_x_2_list[i])
a_2=((n_2*sum(xy_2)-sum(test_x_2_list)*sum(test_y_2_list))/(n_2*sum(x_sqr_2)-(sum(test_x_2_list)*sum(test_x_2_list))))
b_2=(sum(test_y_2_list)-a_2*sum(test_x_2_list))/n_2
for i in range(len(test_x_2_list)):
    y_2.append(a_2*test_x_2_list[i]+b_2)

for i in range(len(test_x_3_list)):
    xy_3.append(test_x_3_list[i]*test_y_3_list[i])
    x_sqr_3.append(test_x_3_list[i]*test_x_3_list[i])
a_3=((n_3*sum(xy_3)-sum(test_x_3_list)*sum(test_y_3_list))/(n_3*sum(x_sqr_3)-(sum(test_x_3_list)*sum(test_x_3_list))))
b_3=(sum(test_y_3_list)-a_3*sum(test_x_3_list))/n_3
for i in range(len(test_x_3_list)):
    y_3.append(a_3*test_x_3_list[i]+b_3)

print (a_1, b_1)
print (a_2, b_2)
print (a_3, b_3)
#print(test_x_1_list[0])
#print(test_y_1_list[0])
#print(test_x_1_list[0]*test_y_1_list[0])
#print(xy)
#print(x_sqr)
#print(a_1)
#print(b_1)

#print(a_2)
#print(b_2)

#print(a_3)
#print(b_3)
#print(y_1)

#D_x_1=0
#D_x_2=0
#D_x_3=0
#D_y_1=0
#D_y_2=0
#D_y_3=0

#binsvalues_x_1=list()
#binsvalues_y_1=list()
#binsvalues_xy_1=list()
#binsvalues_xx_1=list()
#binsvalues_yy_1=list()
mult=list()

#for j in range (0,100,1):
    #binsvalues_y_1.append((j+1)/100)
    #py.append(test_y_1_list[j]/5000)

def mean(x):
    return sum(x)/len(x)

def cov(x,y):
    mult.clear()
    for i in range(len(x)):
        mult.append(x[i]*y[i])
    return mean(mult)-mean(x)*mean(y)

#for i in range(len(test_x_1_list)):
    #binsvalues_xy_1.append(test_x_1_list[i]*test_y_1_list[i])
    #binsvalues_xx_1.append(test_x_1_list[i]*test_x_1_list[i])
    #binsvalues_yy_1.append(test_y_1_list[i]*test_y_1_list[i])
    #binsvalues_x_1.append((i+1)/100)
    #px.append(test_x_1_list[i]/5000)
    #for j in range (0,100,1):
        #binsvalues_xy_1.append(binsvalues_x_1[i]*binsvalues_y_1[j])

#E_x_1=sum(test_x_1_list)/len(test_x_1_list)
#E_x_1=(1/100)*sum(binsvalues_x_1)
#E_y_1=sum(test_y_1_list)/len(test_y_1_list)
#E_y_1=(1/100)*sum(binsvalues_y_1)
#E_xy_1=sum(binsvalues_xy_1)/len(binsvalues_xy_1)
#E_xx_1=sum(binsvalues_xx_1)/len(binsvalues_xx_1)
#E_yy_1=sum(binsvalues_yy_1)/len(binsvalues_yy_1)
#E_xy_1=(1/100)*(1/100)*sum(binsvalues_xy_1)
cov_1=cov(test_x_1_list,test_y_1_list)
cov_1_xx=cov(test_x_1_list,test_x_1_list)
cov_1_yy=cov(test_y_1_list,test_y_1_list)
#for i in range (0,100,1):
    #D_x_1+=(1/100)*(binsvalues_x_1[i]-E_x_1)*(binsvalues_x_1[i]-E_x_1)
    #D_y_1+=(1/100)*(binsvalues_y_1[i]-E_y_1)*(binsvalues_y_1[i]-E_y_1)

cor_1=cov_1/(math.sqrt(cov_1_xx)*math.sqrt(cov_1_yy))

#print(sum(binsvalues_x_1))
#print(sum(binsvalues_y_1))
#print(E_x_1)
#print(E_y_1)
#print(binsvalues_x_1)
#print(binsvalues_y_1)
#print(binsvalues_xy)
print("Ковариация 1 = ", cov_1)
#print(cov_1_xx, cov_1_yy)
print("Корреляционный фактор 1 = ", cor_1, "\n")

#binsvalues_x_2=list()
#binsvalues_y_2=list()
#binsvalues_xy_2=list()

#for j in range (0,100,1):
    #binsvalues_y_2.append(((j+1)/100)*(-5)+0.6)
    #py.append(test_y_1_list[j]/5000)

#for i in range (0,100,1):
    #binsvalues_x_2.append((i+1)/100)
    #px.append(test_x_1_list[i]/5000)
    #for j in range (0,100,1):
        #if i==j:
            #binsvalues_xy_2.append(binsvalues_x_2[i]*binsvalues_y_2[j])


cov_2=cov(test_x_2_list,test_y_2_list)
cov_2_xx=cov(test_x_2_list,test_x_2_list)
cov_2_yy=cov(test_y_2_list,test_y_2_list)

#E_x_2=(1/100)*sum(binsvalues_x_2)
#E_y_2=(1/100)*sum(binsvalues_y_2)
#E_xy_2=(1/(100*math.sqrt(2)))*sum(binsvalues_xy_2)
#cov_2=E_xy_2-E_x_2*E_y_2

#for i in range (0,100,1):
    #D_x_2+=(1/100)*(binsvalues_x_2[i]-E_x_2)*(binsvalues_x_2[i]-E_x_2)
    #D_y_2+=(1/100)*(binsvalues_y_2[i]-E_y_2)*(binsvalues_y_2[i]-E_y_2)

#cor_2=cov_2/(math.sqrt(D_x_2)*math.sqrt(D_y_2))
cor_2=cov_2/(math.sqrt(cov_2_xx)*math.sqrt(cov_2_yy))

if cor_2<(-1):
    cor_2=-0.999999999999999

#print(binsvalues_x_2)
#print(binsvalues_y_2)
#print(binsvalues_xy)
print("Ковариация 2 = ", cov_2)
#print(cov_2_xx, cov_2_yy)
print("Корреляционный фактор 2 = ", cor_2, "\n")

#m_1=100
#m_2=100
#binsvalues_x_3=list()
#binsvalues_y_3=list()
#binsvalues_z_3=list()
#binsvalues_xy_3=list()
#binsvalues_y_3_filled=list()
#binsvalues_xy_3_filled=[[0]*m_1 for i in range(m_2)]

#for j in range (0,100,1):
    #binsvalues_z_3.append((j+1)/100)
    #binsvalues_y_3.append((j+1)/100)
    #binsvalues_y_3_filled.append(0)
    #py.append(test_y_1_list[j]/5000)

#for i in range (0,100,1):
    #binsvalues_x_3.append((i+1)/100)
    #px.append(test_x_1_list[i]/5000)

#for j in range (0,100,1):
    #if j!=0 and j!=99:
        #for element in test_y_3_list:
            #if element < ((j+1)/100) and element >= (j/100):
                #num=test_y_3_list.index(element)
                #binsvalues_y_3_filled[j]+=1
                #for i in range (0,100,1):
                    #if i!=0 and i!=99:
                        #if test_x_3_list[num] < ((i+1)/100) and test_x_3_list[num] >= (i/100):
                            #binsvalues_xy_3_filled[j][i]+=1
                    #elif i==0:
                        #if test_x_3_list[num] < ((j+1)/100):
                            #binsvalues_xy_3_filled[j][i]+=1
                    #else:
                        #if test_x_3_list[num] >= (j/100):
                            #binsvalues_xy_3_filled[j][i]+=1
    #elif j==0:
        #for element in test_y_3_list:
            #if element < ((j+1)/100):
                #num=test_y_3_list.index(element)
                #binsvalues_y_3_filled[j]+=1
                #for i in range (0,100,1):
                    #if i!=0 and i!=99:
                        #if test_x_3_list[num] < ((i+1)/100) and test_x_3_list[num] >= (i/100):
                            #binsvalues_xy_3_filled[j][i]+=1
                    #elif i==0:
                        #if test_x_3_list[num] < ((j+1)/100):
                            #binsvalues_xy_3_filled[j][i]+=1
                    #else:
                        #if test_x_3_list[num] >= (j/100):
                            #binsvalues_xy_3_filled[j][i]+=1
    #else:
        #for element in test_y_3_list:
            #if element >= (j/100):
                #num=test_y_3_list.index(element)
                #binsvalues_y_3_filled[j]+=1
                #for i in range (0,100,1):
                    #if i!=0 and i!=99:
                        #if test_x_3_list[num] < ((i+1)/100) and test_x_3_list[num] >= (i/100):
                            #binsvalues_xy_3_filled[j][i]+=1
                    #elif i==0:
                        #if test_x_3_list[num] < ((j+1)/100):
                            #binsvalues_xy_3_filled[j][i]+=1
                    #else:
                        #if test_x_3_list[num] >= (j/100):
                            #binsvalues_xy_3_filled[j][i]+=1

#binsvalues_only_x_3=list()

#for i in range (0,100,1):
    #binsvalues_only_x_3.append(0)

#for i in range (0,100,1):
    #for j in range (0,100,1):
        #binsvalues_only_x_3[i]+=binsvalues_xy_3_filled[j][i]

#E_x_3=(1/100)*sum(binsvalues_x_3)
#E_y_3=0
#E_xy_3=0

#summary=0

#for row in binsvalues_xy_3_filled:
    #for elem in row:
        #summary+=elem

#summary=5000

#for i in range (0,100,1):
    #E_y_3+=(binsvalues_y_3_filled[i]/sum(binsvalues_y_3_filled))*((i+1)/100)
    #for j in range (0,100,1):
        #E_xy_3+=(binsvalues_xy_3_filled[j][i]/summary)*((i+1)/100)*((j+1)/100)

cov_3=cov(test_x_3_list,test_y_3_list)
cov_3_xx=cov(test_x_3_list,test_x_3_list)
cov_3_yy=cov(test_y_3_list,test_y_3_list)
#cov_3=E_xy_3-E_x_3*E_y_3

#for i in range (0,100,1):
    #D_x_3+=(1/100)*(binsvalues_x_3[i]-E_x_3)*(binsvalues_x_3[i]-E_x_3)
    #D_y_3+=(binsvalues_y_3_filled[i]/sum(binsvalues_y_3_filled))*(binsvalues_y_3[i]-E_y_3)*(binsvalues_y_3[i]-E_y_3)

#cor_3=cov_3/(math.sqrt(D_x_3)*math.sqrt(D_y_3))
cor_3=cov_3/(math.sqrt(cov_3_xx)*math.sqrt(cov_3_yy))

#print(binsvalues_x_2)
#print(binsvalues_y_2)
#print(binsvalues_xy)
#print(binsvalues_xy_3_filled)
print("Ковариация 3 = ", cov_3)
#print(cov_3_xx, cov_3_yy)
print("Корреляционный фактор 3 = ", cor_3, "\n")

t_1=(abs(cor_1)*math.sqrt(n-2))/(math.sqrt(1-cor_1*cor_1))
t_2=(abs(cor_2)*math.sqrt(n-2))/(math.sqrt(1-cor_2*cor_2))
t_3=(abs(cor_3)*math.sqrt(n-2))/(math.sqrt(1-cor_3*cor_3))

print("t_stat_1 = ", t_1)
print("t_stat_2 = ", t_2)
print("t_stat_3 = ", t_3)

ax1 = sns.jointplot(x=test_x_1_list, y=test_y_1_list, marginal_kws=dict(bins=100, fill=True))
ax1.ax_joint.cla()
ax1.ax_joint.set_xlabel("x", fontsize=25)
ax1.ax_joint.set_ylabel("y", fontsize=25)
ax1.ax_joint.set_title("x, y - псевдослучайные \n\n\n\n\n", fontsize=25)
plt.sca(ax1.ax_joint)
plt.plot(test_x_1_list, y_1, color="k", linewidth=6)
plt.hist2d(test_x_1_list, test_y_1_list, bins=(100, 100), range=[[-0.3, 1.3],[-0.3, 1.3]], cmap=cm.brg);
#plt.colorbar()

ax1 = sns.jointplot(x=test_x_2_list, y=test_y_2_list, marginal_kws=dict(bins=100, fill=True))
ax1.ax_joint.cla()
ax1.ax_joint.set_xlabel("x", fontsize=25)
ax1.ax_joint.set_ylabel("y", fontsize=25)
ax1.ax_joint.set_title("x - псведослучайная, y=-5x+0.6 \n\n\n\n\n", fontsize=25)
plt.sca(ax1.ax_joint)
plt.plot(test_x_2_list, y_2, color="k", linewidth=2)
plt.hist2d(test_x_2_list, test_y_2_list, bins=(100, 100), range=[[-0.3, 1.3],[-4.7, 0.9]], cmap=cm.brg);
#plt.colorbar()

ax1 = sns.jointplot(x=test_x_3_list, y=test_y_3_list, marginal_kws=dict(bins=100, fill=True))
ax1.ax_joint.cla()
ax1.ax_joint.set_xlabel("x", fontsize=25)
ax1.ax_joint.set_ylabel("y", fontsize=25)
ax1.ax_joint.set_title("x, z - псевдослучайные, y=zx \n\n\n\n\n", fontsize=25)
plt.sca(ax1.ax_joint)
plt.plot(test_x_3_list, y_3, color="k", linewidth=6)
plt.hist2d(test_x_3_list, test_y_3_list, bins=(100, 100), range=[[-0.3, 1.3],[-0.3, 1.3]], cmap=cm.brg);
#plt.colorbar()

plt.show()
