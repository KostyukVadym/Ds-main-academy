#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='https://mainacademy.ua/'> <img src='https://mainacademy.ua/wp-content/uploads/2019/03/logo-main.png' alt = 'Хмм, щось з інтернетом'/></a>
# ___
# 
# # Module 7: Supervised learning

# ## Lab work 7
# 
# 

# #### Мета: 
# 
# * навчитися використовувати моделі з учителем

# ### Завдання 1:

# Для цього завдання ми будемо вивчати загальнодоступні дані з [LendingClub.com](www.lendingclub.com). 
# 
# Кредитний клуб пов'язує людей, яким потрібні гроші (позичальники), та людей, які мають гроші (інвесторів). Сподіваємось, як інвестор ви хотіли б інвестувати в людей, які продемонстрували, що вони мають високу ймовірність повернути вам гроші. Ми спробуємо створити модель, яка допоможе це передбачити.

# Кредитний клуб мав [дуже цікавий 2016 рік](https://en.wikipedia.org/wiki/Lending_Club#2016), тож давайте перевіримо деякі їх дані.
# 
# Ми використовуватимемо дані про позики за 2007-2010 роки та намагатимемося класифікувати та прогнозувати, чи повертав позичальник їх повністю.

# Ось що представляють стовпці:
# * `credit.policy`: 1, якщо клієнт відповідає критеріям андеррайтингу кредитів LendingClub.com, і 0 в іншому випадку.
# 
# * `purpose`: мета позики (приймає значення "кредитна_карта", "консолідація боргу", "освітня", "основна_покупка", "малий бізнес" та "всі_інші").
# 
# * `int.rate`: процентна ставка позики, пропорційно (ставка 11% зберігатиметься як 0,11). Позичальникам, які LendingClub.com вважає більш ризикованими, призначаються вищі процентні ставки.
# 
# * `installment`: щомісячні виплати позичальника, якщо позика фінансується.
# 
# * `log.annual.inc`: журнал річного доходу позичальника, який самостійно звітується.
# 
# * `dti`: відношення боргу до доходу позичальника (сума боргу, поділена на річний дохід).
# 
# * `fico`: кредитний рейтинг позичальника FICO.
# 
# * `days.with.cr.line`: кількість днів, коли позичальник мав кредитну лінію.
# 
# * `revol.bal`: кредитний залишок позичальника (сума не виплачена в кінці циклу виставлення рахунків за кредитною карткою).
# 
# * `revol.util`: коефіцієнт використання кридитної лінії позичальника (сума використаної кредитної лінії відносно загальної кількості доступних кредитів).
# 
# * `inq.last.6mths`: кількість запитів позичальників з боку кредиторів за останні 6 місяців.
# 
# * `delinq.2yrs`: кількість разів, протягом яких позичальник прострочував платежі протягом останніх 2 років понад 30 днів.
# 
# * `pub.rec`: кількість публічних записів (заяви про банкрутство, податкова застава або рішення).

# Алгоритм виконання та проміжні завдання:
# 1. Відкрийте файл та виведіть описову статистику
# 
# 2. Створіть дві гістограми по полю fico (перша для значень not.fully.paid=0, друга = 1). Гістограми накладіть одна на одну.
# 
# 3. Відобразіть тенденцію між оцінкою fico та int.rate. Використайте jointplot
# 
# 4. Побудуйте графік lmplot на основі int.rate та fico. Коліром розідліть по полю credit.policy. Розбийте по значеннях цільової функції
# 
# 5. purpose є категорієї, тому варто перетворити за допомогою pd.get_dummies
# 
# 6. Розбийте датасет на тестві та трейнові дані
# 
# 7. Використайте для задачі класифікації такі моделі: дерево рішень, логістична регресія, random forest, XGBoost
# 
# 8. Для кожної з моделей виведіть матрицю, основні метрики
# 
# 8. *Використейте бібіотеку dtreeviz для візуалізації (там, де актуально)
# 
# 9. Використайте ансамбель voting для всіх моделей
# 
# 10. Порівняйте результати моделей та зробіть висновки

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('loan_data.csv')


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.head()


# In[6]:


df.hist(bins=50, figsize=(15,15))
plt.show()


# In[7]:


sns.distplot(not_fully_paid, kde = False, color = 'blue', vertical=False)
plt.title('Not fully paid', fontsize=24)
plt.xlabel('Fico', fontsize=16)

sns.distplot(fully_paid, kde = False, color = 'green', vertical=False)
plt.title('Fully paid', fontsize=24)
plt.xlabel('Fico', fontsize=16)

sns.distplot(not_fully_paid, kde = False, color = 'blue', vertical=False)
sns.distplot(fully_paid, kde = False, color = 'green', vertical=False)
plt.legend(labels=['not_fully_paid','fully_paid'])
plt.xlabel('Fico', fontsize=20)
plt.show()


# In[8]:


sns.jointplot(df['fico'], df['int.rate'], kind='scatter')  
plt.show()  


# In[9]:


sns.jointplot(df['fico'], df['int.rate'], kind='reg')  
plt.show()  


# In[10]:


sns.lmplot(x='int.rate',
           y='fico',
           hue='credit.policy',
           data=df);


# In[11]:


new_purpose = pd.get_dummies(df['purpose'])

new_purpose


# In[12]:


df_1 = df.drop('purpose', 1)
df_1

df_new = pd.concat([df_1, new_purpose], axis = 1, sort = False)
df_new


# In[13]:


X = df_new[
    [ 
    'int.rate',
    'all_other',
    'credit_card', 
    'debt_consolidation',
    'educational',
    'home_improvement',
    'major_purchase',
    'small_business',
    'installment', 
    'log.annual.inc', 
    'dti',
     'fico',
     'days.with.cr.line',
     'revol.bal',
     'revol.util',
     'inq.last.6mths',
     'delinq.2yrs',
     'pub.rec'
    ]
]
y = df_new['credit.policy']


# In[14]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train.shape

X_train

y_train


# In[19]:





# In[15]:




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.35)

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

print('Score train', logmodel.score(X_train,y_train))

print('Score test', logmodel.score(X_test,y_test))


# In[17]:


from sklearn import metrics
import numpy as np

y_predictions = logmodel.predict(X_test)

print('Метрики:')
print(metrics.classification_report(y_test,y_predictions))

print('Confusion matrix:')
print(np.flip(metrics.confusion_matrix(y_test, y_predictions)))


# In[18]:


get_ipython().system('pip install dtreeviz')


# In[19]:


from sklearn import tree

from dtreeviz.trees import *


# In[20]:



from sklearn.tree import DecisionTreeClassifier, plot_tree 
import matplotlib.pyplot as plt

model = DecisionTreeClassifier(min_samples_split=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
model.fit(X_train, y_train)

plt.figure(figsize = (16, 16))
plot_tree(model, feature_names = X.columns, filled = True, rounded = True)
plt.show()

model.score(X_test, y_test)


# In[21]:


from sklearn.datasets import*
from dtreeviz.trees import*

regr = tree.DecisionTreeRegressor(max_depth=2)
#X_train, y_train = df_new.data, df_new.target 
regr.fit(X_train, y_train)

import pandas as pd
data = pd.DataFrame(df_new.data, columns=df_new.feature_names)

data


# In[22]:


regr = tree.DecisionTreeRegressor(max_depth=2)
#X_train, y_train = df_new.data, df_new.target 
regr.fit(X_train, y_train)

#data = pd.DataFrame(df_new.data, columns=df_new.feature_names)
#data['target'] = df_new.target

viz = dtreeviz(regr,
               X_train,
               y_train,
               target_name='price',
               feature_names= ['1', '2', '3'])
              
viz.view()


# In[23]:


from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

viz = dtreeviz(model,
               X_train,
               y_train,
               target_name='price',
               feature_names= df_new.columns)
              
viz.view()

for i in range(1, 100, 10):
    model = RandomForestClassifier(n_estimators=i, random_state=1)
    model.fit(X_train, y_train)
    print(i, model.score(X_test, y_test))

model.score(X_test, y_test)


# In[ ]:


import xgboost as xgb
xgb_class = xgb.XGBClassifier()

xgb_class.fit(X_train, y_train)

xgb_class.score(X_test, y_test)

xgb_class.feature_importances_

xgb.plot_importance(xgb_class)

booster = xgb_class.get_booster()
print(booster.get_dump()[0])

xgb.plot_tree(xgb_class)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Завдання 2:

# Просте завдання на обробку часових рядів.

# Потрібно вивести наступну інформацію: 
# 1. Виведіть список унікальних міст з датасету
# 2. Виведіть дату початку та кінця ведення даних
# 3. Яка середня конценрація $NO_2$ для кожного дня тижня і міста (виведіть таблицю)?
# 4. Яке середнє значення для кожної години (виведіть стовбчикову діаграму)?

# In[7]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt


# In[8]:


df = pd.read_csv('https://raw.githubusercontent.com/dm-fedorov/pandas_basic/master/%D0%B1%D1%8B%D1%81%D1%82%D1%80%D0%BE%D0%B5%20%D0%B2%D0%B2%D0%B5%D0%B4%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%B2%20pandas/data/air_quality_no2_long.csv')


# In[9]:


df.head()


# In[4]:


df.tail()


# In[5]:


len(df['country'].unique())


# In[10]:


df["date.utc"] = pd.to_datetime(df["date.utc"])


# In[11]:


df["date.utc"]


# In[12]:


df["date.utc"].min(), df["date.utc"].max()


# In[14]:


df.groupby([df["date.utc"].dt.weekday, "location"])["value"].mean()


# In[16]:


fig, axs = plt.subplots(figsize=(12, 4))
df.groupby(
                    df["date.utc"].dt.hour)["value"].mean().plot(kind='bar',
                    rot=0,
                    ax=axs)
plt.xlabel("Hour of the day");  
plt.ylabel("$NO_2 (µg/m^3)$");


# In[ ]:




