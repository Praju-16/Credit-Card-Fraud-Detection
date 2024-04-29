#!/usr/bin/env python
# coding: utf-8

# # CREDIT CARD FRAUD DETECTION

# In[1]:


# import the necessary packages 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib import gridspec 
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv("C:\\Users\\AKSHAY\\Downloads\\creditcard.csv\\creditcard.csv") 


# In[3]:


data.head() 


# In[4]:


data.tail() 


# In[5]:


print(data.shape) 


# In[6]:


print(data.describe())


# In[7]:


data.info()


# In[8]:


# Find the null values
data.isnull()


# In[9]:


#count the number of missing values in each column
data.isnull().sum()


# In[10]:


# Determine number of fraud cases in dataset 
fraud = data.loc[data['Class'] == 1] 
valid = data[data['Class'] == 0] 


# In[11]:


len(fraud)


# In[12]:


len(valid)


# In[13]:


sns.relplot(x='Amount',y='Time',hue='Class',data=data)


# In[14]:


#sns.catplot(x='Amount',y='Time',hue='Class',data=data)


# In[15]:


#sns.catplot(x='Amount',y='Time',hue='Class',data=data, kind='box')


# In[16]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[17]:


x = data.iloc[:,:-1]
y = data['Class']


# In[18]:


# Splitting the data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=333)


# In[19]:


x_train.shape


# In[20]:


x_test.shape


# In[21]:


y_train.shape


# In[22]:


y_test.shape


# In[23]:


data.corr()


# # LOGISTIC REGRESION

# In[24]:


# Create model object
logreg = LogisticRegression()
# train the model using fit function
logreg.fit(x_train, y_train)
# predict on x dataset
y_pred = logreg.predict(x_test)


# In[25]:


y_pred =np.array(logreg.predict(x_test))
y = np.array(y_test)


# In[26]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[27]:


print(confusion_matrix(y_test, y_pred))


# In[28]:


print(accuracy_score(y_test, y_pred))


# In[29]:


print(classification_report(y_test, y_pred))


# In[30]:


plt.figure(figsize=(7,5))
sns.scatterplot(x=data["Time"], y=data["Amount"], hue=data["Class"], size=data["Class"], sizes=(40, 8), marker="+")


# In[31]:


plt.figure(figsize=(8,5))
plot = sns.distplot(a=data["Time"], kde=True, color='purple')
plot.set(xlabel ='Time', ylabel ='Frequency')
plt.show()


# In[32]:


plt.figure(figsize=(18,8))
data = data[data["Class"] == 1]
sns.distplot(a=data["V16"], kde="True")
sns.distplot(a=data["V17"], kde="True")
sns.distplot(a=data["V18"], kde="True")


# In[33]:


data.Class.value_counts()
sns.countplot("Class",data=data)


# In[ ]:




