#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


X = pd.read_csv('./Training Data/Linear_X_Train.csv')
Y = pd.read_csv('./Training Data/Linear_Y_Train.csv')


# In[5]:


theta = np.load("ThetaList.npy")
T0 = theta[:,0]
T1 = theta[:,1]


# In[6]:


# ion is for interactive mode in matplotlib 
plt.ion()
for i in range(0,50,3):
    y_ = T1[i]*X + T0
    # points
    plt.scatter(X,Y)
    # line
    plt.plot(X,y_,'red')
    plt.pause(1) # pause graph for 1 second
    plt.clf() #Destroy the last object


# In[ ]:




