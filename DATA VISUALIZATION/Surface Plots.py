#!/usr/bin/env python
# coding: utf-8

# # Surface Plots are used to visualize , loss functions in machine and deep learning.It also visualises state and state value functions in Reinforcement learning.

# In[14]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


# In[15]:


a=np.array([1,2,3])
b=np.array([4,5,6,7])
# meshgrid , will repeat a b number of times rowise in a and then in b it would repeat b a number of times columnwise so 
# dimentionality of both a and b after meshgrid would be 4*3.This meshgrid is also going to give us the coordinates of the plane
# that we are going to make
a,b = np.meshgrid(a,b)
print(a)
print()
print(b)


# In[16]:


fig = plt.figure()
axes = fig.gca(projection = '3d')
axes.plot_surface(a,b,a+b,cmap = 'coolwarm') # a will go on the x axis , b on y and a+b on the z axis  
plt.show # inc loolmap red denotes the value with high value and blue with low value.


# In[17]:


# NOw trying same thing on different values where z = a^2+b^2
a=np.arange(-1,1,0.02)
b=np.arange(-1,1,0.02)
a,b = np.meshgrid(a,b)
fig = plt.figure()
axes = fig.gca(projection = '3d')
axes.plot_surface(a,b,a**2+b**2,cmap = 'coolwarm') # a will go on the x axis , b on y and a+b on the z axis  
plt.show


# # We can see another plot named as contour plot which separates our bowl in plane

# In[13]:


fig = plt.figure()
axes = fig.gca(projection = '3d')
axes.contour(a,b,a**2+b**2,cmap = 'rainbow')
plt.show()


# In[ ]:




