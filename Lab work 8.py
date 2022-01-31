#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='https://mainacademy.ua/'> <img src='https://mainacademy.ua/wp-content/uploads/2019/03/logo-main.png' alt = 'Хмм, щось з інтернетом'/></a>
# ___
# 
# # Module 8: Unsupervised learning

# ## Lab work 8
# 
# 

# #### Мета: 
# 
# * навчитися використовувати моделі без учителем

# ### Завдання 1:

# Опрацюйте файл `Groceries.csv`. Використовуючи алгоритм асоціативних зв'язків, знайдіть:
# - зробіть описову статистику даного файлу
# - ведіть топ-3 продукти, що продаються
# - виведіть топ-2 пари продуктів, що продаються
# - придумайте, як отриману інформацію можна використатти (опишіть в 3-4 реченнях)

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('groceries.csv')


# In[3]:


df.info()


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.head()


# In[ ]:


transaction = []
for i in range(df.shape[0]):
    transaction.append([str(df.values[i,j]) for j in range(df.shape[1])])
    
transaction = np.array(transaction)
transaction


# In[ ]:


te = TransactionEncoder()
te_ary = te.fit(transaction).transform(transaction)
dataset = pd.DataFrame(te_ary, columns=te.columns_)
dataset


# In[ ]:


def encode_units(x):
    if x == False:
        return 0 
    if x == True:
        return 1
    
dataset = dataset.applymap(encode_units)
dataset.head(10)


# In[ ]:


frequent_itemsets = apriori(dataset, min_support=0.01, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets


# In[ ]:


frequent_itemsets[ (frequent_itemsets['length'] == 2) &
                   (frequent_itemsets['support'] >= 0.05) ]


# In[ ]:


frequent_itemsets[ (frequent_itemsets['length'] == 3) ].head()


# ### Завдання 2:

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Завантажте файл https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python
#  
# - Спробуйте провести кластеризацію, тобто виділити цільові групи, для яких можна впроваджувати певні маркетингові акції
# - Опишіть отримані результати

# In[7]:


df = pd.read_csv('Mall_Customers.csv')


# In[8]:


df.info()


# In[9]:


df.head()


# In[10]:


X = df.iloc[:, [3, 4]].values


# In[11]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[12]:


kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)


# In[13]:


plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[ ]:




