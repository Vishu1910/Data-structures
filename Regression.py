
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston


# In[10]:


boston=load_boston()
print(boston.DESCR)


# In[11]:


dataset = boston.data
for name,index in enumerate(boston.feature_names):
    print(index,name)


# In[12]:


data=dataset[:,12].reshape(-1,1)


# In[13]:


np.shape(dataset)


# In[14]:


target=boston.target.reshape(-1,1)

np.shape(target)
# In[9]:


np.shape(target)


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data,target,color='blue')
plt.xlabel('Lower income population')
plt.ylabel('Cost of House')
plt.show()


# In[22]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(data,target)


# In[23]:


pred=reg.predict(data)


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data,target,color='blue')
plt.plot(data,pred,color='yellow')
plt.xlabel('Lower income population')
plt.ylabel("Cost of House")
plt.show()


# In[24]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# In[27]:


model= make_pipeline(PolynomialFeatures(3),reg)


# In[28]:


model.fit(data,target)


# In[29]:


pred=model.predict(data)


# In[30]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data,target,color='Blue')
plt.plot(data,pred,color='Yellow')
plt.xlabel('Low Income Population')
plt.ylabel('Cost of House')
plt.show()


# In[31]:


from sklearn.metrics import r2_score


# In[32]:


r2_score(pred,target)

