#!/usr/bin/env python
# coding: utf-8

# # Week 2 lecture notebook

# ## Outline
# 
# [Missing values](#missing-values)
# 
# [Decision tree classifier](#decision-tree)
# 
# [Apply a mask](#mask)
# 
# [Imputation](#imputation)

# <a name="missing-values"></a>
# ## Missing values

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.DataFrame({"feature_1": [0.1,np.NaN,np.NaN,0.4],
                   "feature_2": [1.1,2.2,np.NaN,np.NaN]
                  })
df


# ### Check if each value is missing

# In[3]:


df.isnull()


# ### Check if any values in a row are true
# 

# In[4]:


df_booleans = pd.DataFrame({"col_1": [True,True,False],
                            "col_2": [True,False,False]
                           })
df_booleans


# - If we use pandas.DataFrame.any(), it checks if at least one value in a column is `True`, and if so, returns `True`.
# - If all rows are `False`, then it returns `False` for that column

# In[5]:


df_booleans.any()


# - Setting the axis to zero also checks if any item in a column is `True`

# In[6]:


df_booleans.any(axis=0)


# - Setting the axis to `1` checks if any item in a **row** is `True`, and if so, returns true
# - Similarily only when all values in a row are `False`, the function returns `False`.

# In[7]:


df_booleans.any(axis=1)


# ### Sum booleans

# In[8]:


series_booleans = pd.Series([True,True,False])
series_booleans


# - When applying `sum` to a series (or list) of booleans, the `sum` function treats `True` as 1 and `False` as zero.

# In[9]:


sum(series_booleans)


# You will make use of these functions in this week's assignment!

# ### This is the end of this practice section.
# 
# Please continue on with the lecture videos!
# 
# ---

# <a name="decision-tree"></a>
# ## Decision Tree Classifier
# 

# In[10]:


import pandas as pd


# In[11]:


X = pd.DataFrame({"feature_1":[0,1,2,3]})
y = pd.Series([0,0,1,1])


# In[12]:


X


# In[13]:


y


# In[14]:


from sklearn.tree import DecisionTreeClassifier


# In[15]:


dt = DecisionTreeClassifier()
dt


# In[16]:


dt.fit(X,y)


# ### Set tree parameters

# In[17]:


dt = DecisionTreeClassifier(criterion='entropy',
                            max_depth=10,
                            min_samples_split=2
                           )
dt


# ### Set parameters using a dictionary
# 
# - In Python, we can use a dictionary to set parameters of a function.
# - We can define the name of the parameter as the 'key', and the value of that parameter as the 'value' for each key-value pair of the dictionary.

# In[18]:


tree_parameters = {'criterion': 'entropy',
                   'max_depth': 10,
                   'min_samples_split': 2
                  }


# - We can pass in the dictionary and use `**` to 'unpack' that dictionary's key-value pairs as parameter values for the function.

# In[19]:


dt = DecisionTreeClassifier(**tree_parameters)
dt


# ### This is the end of this practice section.
# 
# Please continue on with the lecture videos!
# 
# ---

# <a name="mask"></a>
# ## Apply a mask
# 
# Use a 'mask' to filter data of a dataframe

# In[20]:


import pandas as pd


# In[21]:


df = pd.DataFrame({"feature_1": [0,1,2,3,4]})
df


# In[22]:


mask = df["feature_1"] >= 3
mask


# In[23]:


df[mask]


# ### Combining comparison operators
# 
# You'll want to be careful when combining more than one comparison operator, to avoid errors.
# - Using the `and` operator on a series will result in a `ValueError`, because it's 

# In[24]:


df["feature_1"] >=2


# In[25]:


df["feature_1" ] <=3


# In[26]:


# NOTE: This will result in a ValueError
df["feature_1"] >=2 and df["feature_1" ] <=3


# ### How to combine two logical operators for Series
# What we want is to look at the same row of each of the two series, and compare each pair of items, one row at a time. To do this, use:
# - the `&` operator instead of `and`
# - the `|` operator instead of `or`.
# - Also, you'll need to surround each comparison with parenthese `(...)`

# In[27]:


# This will compare the series, one row at a time
(df["feature_1"] >=2) & (df["feature_1" ] <=3)


# ### This is the end of this practice section.
# 
# Please continue on with the lecture videos!
# 
# ---

# <a name="imputation"></a>
# ## Imputation
# 
# We will use imputation functions provided by scikit-learn.  See the scikit-learn [documentation on imputation](https://scikit-learn.org/stable/modules/impute.html#iterative-imputer)

# In[28]:


import pandas as pd
import numpy as np


# In[29]:


df = pd.DataFrame({"feature_1": [0,1,2,3,4,5,6,7,8,9,10],
                   "feature_2": [0,np.NaN,20,30,40,50,60,70,80,np.NaN,100],
                  })
df


# ### Mean imputation

# In[30]:


from sklearn.impute import SimpleImputer


# In[31]:


mean_imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
mean_imputer


# In[32]:


mean_imputer.fit(df)


# In[33]:


nparray_imputed_mean = mean_imputer.transform(df)
nparray_imputed_mean


# Notice how the missing values are replaced with `50` in both cases.

# ### Regression Imputation

# In[34]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# In[35]:


reg_imputer = IterativeImputer()
reg_imputer


# In[36]:


reg_imputer.fit(df)


# In[37]:


nparray_imputed_reg = reg_imputer.transform(df)
nparray_imputed_reg


# Notice how the filled in values are replaced with `10` and `90` when using regression imputation. The imputation assumed a linear relationship between feature 1 and feature 2.

# ### This is the end of this practice section.
# 
# Please continue on with the lecture videos!
# 
# ---

# In[ ]:




