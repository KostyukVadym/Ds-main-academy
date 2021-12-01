#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='https://mainacademy.ua/'> <img src='https://mainacademy.ua/wp-content/uploads/2019/03/logo-main.png' alt = 'Хмм, щось з інтернетом'/></a>
# ___
# 
# # Module 5: Data analysis with NumPy and Pandas

# ## Lab work 5.2
# 
# 

# #### Мета: 
# 
# * навчитися працювати із бібліотекою Pandas в Python.

# ### Завдання:

# In[ ]:


import pandas as pd
import numpy as np


# Вивести версію та конфігурацію бібліотеки:

# In[1]:


import pandas as pd
pd.__version__


# In[2]:


import pandas as pd
df = pd.read_csv('Ecommercepurchases.txt')

df.head()


# Скільки рядків і стовпців в наборі даних:

# In[3]:


df.shape


# Перевірити, чи в наборі даних містяться порожні значення:

# In[4]:


df[df.notnull()].count()


# Яка середня ціна закупки (Purchase Price):

# In[5]:


df['Purchase Price'].mean()


# Скільки людей користуються англійською мовою "en" на веб-сайті:

# In[6]:


df[df['Language'] == 'en'].shape[0]


# Скільки людей має посаду «Lawyer»?

# In[7]:


df[df['Job'] == 'Lawyer'].shape[0]


# Скільки людей зробило покупку вранці та скільки людей зробило покупку після обіду?

# In[8]:


a = df['AM or PM']
counts = a.value_counts()
counts['AM'], counts['PM']


# Які 5 найпоширеніших назв вакансій?

# In[9]:


df['Job'].value_counts().head(5)


# Хтось здійснив покупку, яка надійшла від Lot: "90 Wt", та якою була ціна придбання для цієї транзакції?

# In[10]:


df[df['Lot'] == '94 vE']['Purchase Price']


# Яка електронна адреса особи з таким номером кредитної картки: 4926535242672853 ?

# In[11]:


df[df['Credit Card'] == 4926535242672853]['Email']


# Скільки людей використовує American Express  і здійснили покупку на суму понад 100 доларів?

# In[18]:


a = df[df['Purchase Price'].values > 100].count()[0]
print('кількість людей, які використовують картку American Express, здійснили покупку на суму понад 100 доларів: ', a)


# Скільки людей мають кредитну картку, термін дії якої закінчується в 2025 році?

# In[20]:


qnty_cc_25 = len(df[df['CC Exp Date'].str.contains('/25')])

print('кількість людей, які мають кредитну картку, термін дії якої закінчується в 2025 році: ', qnty_cc_25)


# Які найкращі 5 найпопулярніших постачальників / хостів електронної пошти (наприклад, gmail.com, yahoo.com тощо ...).

# In[69]:


def mail(email):
    l = email.split('@')
    return l[1]

df['Domen'] = df['Email'].map(mail)


df['Domen'].value_counts().head(5)





# Виведіть зведену таблицю по браузерах(Browser Info), посаді(Job), та кількості транзакцій :

# In[23]:


df_2 = df[['Browser Info', 'Job']]
df_2
print('Кількість транзакцій: ', len(df_2))


# Створіть нову колонку "Actual price", яка утворюється із "Purchase Price" та націнки за принципом:
# - якщо "Purchase Price" > 50, націнка 20%
# - якщо "Purchase Price" > 100, націнка 10%
# - в інших випадках націнка 30%
#    

# In[26]:



def Actual_price(df):
    if df['Purchase Price'] >50:
        return df['Purchase Price'] * 1.2
    if df['Purchase Price'] >100:
        return df['Purchase Price'] * 1.1
    return df['Purchase Price'] * 1.3
        

df['Actual_price'] = df.apply(Actual_price, axis = 1)
df


# In[ ]:





# Зробіть рангування набору даних по "Language" та "Actual price" в порядку спадання ціни.
# 
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rank.html

# In[28]:


df["Rank"] = df[['Language', 'Actual_price']].apply(tuple,axis=1).rank(method='dense',ascending=False).astype(int)
print(df["Rank"])


# Колонку "Language" (категоріальна змінна) "закодуйте", тобто утворити індикаторні колоник. В наборі не повинна залишитися колонка "Language".
# 
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rank.html

# In[29]:


pd.get_dummies(df.Language)


# Кінцевий набір даних збережіть у файл з розширенням csv.

# In[1]:


df.to_csv('Ecommerce Purchases_new.csv')


# In[ ]:




