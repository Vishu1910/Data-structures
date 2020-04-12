
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


data= pd.read_csv("mnist_test.csv")


# In[14]:


data.head()


# In[17]:


a = data.iloc[4,1:].values


# In[19]:


#a = a.reshape(28,28).astype('uint8')
plt.imshow(a)


# In[20]:


df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]


# In[22]:


x_train,x_test, y_train, y_test = train_test_split(df_x,df_y, test_size=0.2, random_state=4)


# In[24]:


y_train.head()


# In[25]:


rf = RandomForestClassifier(n_estimators=100)


# In[26]:


rf.fit(x_train, y_train)


# In[27]:


pred = rf.predict(x_test)


# In[28]:


pred


# In[29]:


a = y_test.values
count = 0
for i in range(len(pred)):
    if pred[i] == a[i]:
        count=count+1


# In[30]:


count


# In[31]:


len(pred)


# In[33]:


#accuracy 
1886/2000

