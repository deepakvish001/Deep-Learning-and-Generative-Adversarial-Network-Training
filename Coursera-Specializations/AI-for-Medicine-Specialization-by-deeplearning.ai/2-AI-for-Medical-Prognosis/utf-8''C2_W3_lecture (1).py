#!/usr/bin/env python
# coding: utf-8

# # AI4M Course 2 Week 3 lecture notebook

# ## Outline
# 
# [Count patients](#count-patients)
# 
# [Kaplan-Meier](#kaplan-meier)

# <a name="count-patients"></a>
# ## Count patients

# In[1]:


import numpy as np
import pandas as pd


# We'll work with data where:
# - Time: days after a disease is diagnosed and the patient either dies or left the hospital's supervision.
# - Event: 
#     - 1 if the patient died
#     - 0 if the patient was not observed to die beyond the given 'Time' (their data is censored)
#     
# Notice that these are the same numbers that you see in the lecture video about estimating survival.

# In[2]:


df = pd.DataFrame({'Time': [10,8,60,20,12,30,15],
                   'Event': [1,0,1,1,0,1,0]
                  })
df


# ### Count patients 

# ### Count number of censored patients

# In[3]:


df['Event'] == 0


# Patient 1, 4 and 6 were censored.

# - Count how many patient records were censored
# 
# When we sum a series of booleans, `True` is treated as 1 and `False` is treated as 0.

# In[4]:


sum(df['Event'] == 0)


# ### Count number of patients who definitely survived past time t
# 
# This assumes that any patient who was censored died at the time of being censored ( **died immediately**).
# 
# If a patient survived past time `t`:
# - Their `Time` of event should be greater than `t`.  
# - Notice that they can have an `Event` of either 1 or 0.  What matters is their `Time` value.

# In[5]:


t = 25
df['Time'] > t


# In[6]:


sum(df['Time'] > t)


# ### Count the number of patients who may have survived past t
# 
# This assumes that censored patients **never die**.
# - The patient is censored at any time and we assume that they live forever.
# - The patient died (`Event` is 1) but after time `t`

# In[7]:


t = 25
(df['Time'] > t) | (df['Event'] == 0)


# In[8]:


sum( (df['Time'] > t) | (df['Event'] == 0) )


# ### Count number of patients who were not censored before time t

# If patient was not censored before time `t`:
# - They either had an event (death) before `t`, at `t`, or after `t` (any time)
# - Or, their `Time` occurs after time `t` (they may have either died or been censored at a later time after `t`)

# In[9]:


t = 25
(df['Event'] == 1) | (df['Time'] > t)


# In[10]:


sum( (df['Event'] == 1) | (df['Time'] > t) )


# <a name="kaplan-meier"></a>
# ## Kaplan-Meier
# 
# The Kaplan Meier estimate of survival probability is:
# 
# $$
# S(t) = \prod_{t_i \leq t} (1 - \frac{d_i}{n_i})
# $$
# 
# - $t_i$ are the events observed in the dataset 
# - $d_i$ is the number of deaths at time $t_i$
# - $n_i$ is the number of people who we know have survived up to time $t_i$.
# 

# In[11]:


import numpy as np
import pandas as pd


# In[12]:


df = pd.DataFrame({'Time': [3,3,2,2],
                   'Event': [0,1,0,1]
                  })
df


# ### Find those who survived up to time $t_i$
# 
# If they survived up to time $t_i$, 
# - Their `Time` is either greater than $t_i$
# - Or, their `Time` can be equal to $t_i$

# In[13]:


t_i = 2
df['Time'] >= t_i


# You can use this to help you calculate $n_i$

# ### Find those who died at time $t_i$
# 
# - If they died at $t_i$:
# - Their `Event` value is 1.  
# - Also, their `Time` should be equal to $t_i$

# In[14]:


t_i = 2
(df['Event'] == 1) & (df['Time'] == t_i)


# You can use this to help you calculate $d_i$
# 
# You'll implement Kaplan Meier in this week's assignment!
