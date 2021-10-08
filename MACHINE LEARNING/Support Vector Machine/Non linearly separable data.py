#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import make_circles
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# In[16]:


X,Y = make_circles(n_samples=500,noise=0.02)


# In[17]:


print(X.shape," ",Y.shape)


# In[18]:


plt.figure(figsize=(6,6))
plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()


# In[19]:


def phi(X):
    """Non linear transformation"""
    X1 = X[:,0]
    X2 = X[:,1]
    X3 = X1**2 + X2**2
    
    X_ = np.zeros((X.shape[0],3))
    print(X_.shape)
    
    X_[:,:-1] = X
    X_[:,-1] = X3
    return X_


# In[20]:


X_ = phi(X)


# In[21]:


print(X[:3,:])
print(X_[:3,:])


# In[41]:


def plot3d(X,show=True):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111,projection='3d')
    X1 = X[:,0]
    X2 = X[:,1]
    X3 = X[:,2]
    
    ax.scatter(X1,X2,X3,zdir='z',s=20,c=Y,depthshade=True)
    if show==True:
        plt.show()
    return ax


# In[42]:


plot3d(X_)


# # Logistic Classifier

# In[43]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# In[44]:


lr = LogisticRegression()


# In[45]:


acc = cross_val_score(lr,X,Y,cv=5).mean()
print("Accuracy X(2d) is %.4f"%(acc*100))


# # 46 % accuracy is not a good accuracy so we would try to try the logistic classifier on our new datset which is 3d

# In[46]:


acc = cross_val_score(lr,X_,Y,cv=5).mean()
print("Accuracy X(3d) is %.4f"%(acc*100))


# ## Accuracy is now 100% so also lets visualize the decision surface

# In[47]:


lr.fit(X_,Y)


# In[48]:


wts = lr.coef_
print(wts)


# In[49]:


bias = lr.intercept_ # this would give us the intercept
print(bias)


# ### The plane we have will have the equation as ax+by+cz+d=0 ==> z = -(ax+by+d)/c where a= w[0] , b = w[1] , c=w[2]

# In[50]:


xx,yy= np.meshgrid(range(-2,2),range(-2,2))
print(xx,"\n",yy)


# In[51]:


z = -(wts[0][0]*xx + wts[0,1]*yy + bias)/wts[0][2]
print(z)


# In[55]:


ax = plot3d(X_,False) # plot3d() return an object which is stored as ax i.e axis
ax.plot_surface(xx,yy,z,alpha=0.2)
plt.show()


# In[ ]:




