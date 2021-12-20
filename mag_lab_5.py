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

x=list()
u=list()
s=list()
u_k=list()
s_k=list()
u_t_1=list()
u_t_2=list()
s_k,u_k=list(),list()

n=100000
for i in range(n):
    x.append(random.random())
    u.append(random.random())
    s.append(random.random())

for i in range(n):
    u_t_1.append(u[i]+0.5)
    u_t_2.append(0.5*u[i]+0.5)
    #u_t_2.append(u[i])
    if (s[i]<=u_t_1[i])&(s[i]>=u_t_2[i])&(s[i]<=1):
        s_k.append(s[i])
        u_k.append(u[i])

d_f= dict()
d_f['x']= x
data = pd.DataFrame(d_f)
#print(data)


plt.subplot(2, 2, 1)
plt.title("Basic distribution")
plt.hist(x, bins=(100))

plt.subplot(2, 2, 2)
plt.title("The random")
plt.plot(u, s, '.', color="orange")
plt.plot(u_k, s_k, '.', color="tab:green")
plt.axis([-1, 2, -0.5, 1.5])


plt.show()
