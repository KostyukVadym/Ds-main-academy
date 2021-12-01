#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='https://mainacademy.ua/'> <img src='https://mainacademy.ua/wp-content/uploads/2019/03/logo-main.png' alt = 'Хмм, щось з інтернетом'/></a>
# ___
# 
# # Module 6: Basics of data visualization

# ## Lab work 6
# 
# 

# #### Мета: 
# 
# * навчитися візуалізовувати дані в Python

# ### Завдання:

# In[1]:


import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


np.random.seed(0)

df = pd.DataFrame(data={'a':np.random.randint(0, 100, 30),
                        'b':np.random.randint(0, 100, 30),
                        'c':np.random.randint(0, 100, 30)})
df.head()


# Створити візуалізацію, аналогічно рисунку 
#  - перші 3 графіки візуалізувати, викорстовуючи значення із df
#  - останній це пряма пропорційність

# In[5]:


fig,  axes = plt.subplots(2, 2, figsize=(14, 8))

x = np.linspace(0, 5, 30)
y = x

axes[0][0].plot(df['a'])
axes[0][1].plot(df['b'])
axes[1][0].plot(df['c'])
axes[1][1].plot(x, y)


# Створити візуалізацію, аналогічно рисунку 

# In[6]:


plt.figure(figsize=(15, 7))
a = plt.plot(df['a'])


# Створити візуалізацію, аналогічно рисунку 
# - використовуйте колонки `a` та `b`

# In[9]:


fig,  axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(df['a']) 

axes[1].plot(df['b'])


# Створити візуалізацію, аналогічно рисунку 
# - використовуйте колонки `a` та `b`

# In[10]:


plt.figure(figsize=(15, 7))
a = plt.plot(df['a'])
a = plt.plot(df['b'])


# Створити візуалізацію, аналогічно рисунку 
# - використовуйте колонки `a` та `b`
# - задайте стиль 'darkgrid' за допомогою команди `sns.set_style`

# In[11]:


sns.set()
plt.figure(figsize=(15, 7))

a = plt.plot(df['a'])
b = plt.plot(df['b'])


# Створити візуалізацію, аналогічно рисунку 
# - для колонки `a` використайте червоний колір та лінію формату `-.`
# 
# - для колонки `b` використайте помаранчевий колір та товщину `10`
# 
# - для колонки `c` використайте жовтий колір та товщину `1` і маркер `o`
# 

# In[13]:


sns.set()
plt.figure(figsize=(16, 8))

a = plt.plot(df['a'],  linestyle = '-.',
        color = 'red')
b = plt.plot(df['b'],  linewidth = 7,
        color = 'orange')
c = plt.plot(df['c'],
        marker = 'o',
        color = 'yellow')


# Створити візуалізацію, аналогічно рисунку 
# - і не забудьте про легенду :)

# In[12]:


fig,  axes = plt.subplots(3, 1, figsize=(14, 9))

axes[0].plot(df['a'], label = 'line(a)') 
axes[0].legend(loc = 'lower right')

axes[1].plot(df['b'], label = 'line(b)')
axes[1].legend(loc = 'center left')
axes[2].plot(df['c'])


# Створити візуалізацію, аналогічно рисунку 
# - використайте томатний колір та відстань між стовбцями 0.5

# In[10]:





# Створити візуалізацію, аналогічно рисунку 
# - добавте всі підписи та правильний маркер

# In[12]:




