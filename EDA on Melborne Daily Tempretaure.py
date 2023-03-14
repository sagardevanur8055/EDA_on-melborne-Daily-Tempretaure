#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[50]:


df = pd.read_csv('D:\Projects\Minimum_Tempretaure_Data.csv', error_bad_lines=False, parse_dates=[0])
df.head()


# In[51]:


df.shape


# In[52]:


df.info()


# In[53]:


df.isnull() .sum()


# In[54]:


df = df.rename(columns={'Daily minimum temperatures in Melbourne, Australia, 1981-1990' : 'Temp'})
df.head()


# In[55]:


temperature_values = df['Temp'].unique()
temperature_values


# In[56]:


notdigit = df[~df['Temp'].str[0].str.isdigit()]
notdigit


# In[57]:


df['Temp'] = df['Temp'].apply(lambda x:x.replace('?', ''))
df['Temp'] = df['Temp'].astype(float)


# In[58]:


df.info()


# In[59]:


df.describe()


# In[60]:


plt.figure(figsize=(12,6))
sns.lineplot(x=df['Date'], y=df['Temp'])
plt.grid(axis='x')
plt.grid(axis='y', alpha=0.3)
plt.show()


# In[61]:


def get_trend(timeseries, deg=3):
  x = list(range(len(timeseries)))
  y = timeseries.values
  coef = np.polyfit(x, y, deg)
  trend = np.poly1d(coef)(x)
  return pd.Series(data=trend, index = timeseries.index)

df['Trend'] = get_trend(df['Temp'])
df.head()


# In[62]:


plt.figure(figsize=(14,6))
sns.lineplot(x=df['Date'], y=df['Temp'], alpha=0.5, label='Temp')
sns.lineplot(x=df['Date'], y=df['Trend'], label='Trend')
plt.grid(axis='x')
plt.legend()
plt.show()


# In[63]:


df2 = df[df['Date'].dt.year == 1981].iloc[:, :2]
df2.head()


# In[64]:


plt.figure(figsize=(12,6))
sns.lineplot(x=df2['Date'], y=df2['Temp'])
plt.grid(axis='x')
plt.grid(axis='y', alpha=0.3)
plt.show()


# In[65]:


def get_trend(timeseries, deg=3):
  x = list(range(len(timeseries)))
  y = timeseries.values
  coef = np.polyfit(x, y, deg)
  trend = np.poly1d(coef)(x)
  return pd.Series(data=trend, index = timeseries.index)

df2['Trend'] = get_trend(df2['Temp'])
df2.head()


# In[67]:


plt.figure(figsize=(14,6))
sns.lineplot(x=df2['Date'], y=df2['Temp'], alpha=0.5, label='Temp')
sns.lineplot(x=df2['Date'], y=df2['Trend'], label='Trend')
plt.grid(axis='x')
plt.legend()
plt.show()


# In[68]:


bymonth = df2.groupby(df2['Date'].dt.month_name(), sort=False).mean()
bymonth = bymonth.iloc[:, :1]
bymonth


# In[70]:


plt.figure(figsize=(14, 7))
bar = sns.barplot(bymonth.index, 'Temp', data=bymonth)
bar.set_ylabel('Mean by month')
bar.set_xlabel('Month in 1981')

plt.show()


# In[ ]:




