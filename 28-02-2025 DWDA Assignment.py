#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


df = pd.read_excel("D:\\DWDA\\crop yield data sheet (1).xlsx")
df.head()


# In[8]:


#Data Preprocessing


# In[9]:


#Checking the shape of the dataset
df.shape


# In[10]:


#Checking the data types of the columns
df.dtypes


# In[13]:


df['Temperatue'].unique()


# In[15]:


#Dropping ":" from the temperature column
df = df[df['Temperatue'] != ':']


# In[21]:


#converting the Temperatue column to float
df['Temperatue'] = df['Temperatue'].astype(float)


# In[23]:


#Checking for null values
df.isnull().sum()


# In[24]:


#replacing missing values with median
columns = [df.columns]
for col in columns:
    df[col] = df[col].fillna(df[col].median())


# In[25]:


df.describe()


# In[26]:


df.head()


# In[27]:


#Rainfall Distribution


# In[28]:


sns.histplot(x = "Rain Fall (mm)", data = df, kde = True)


# In[29]:


#Fertilizer Distribution


# In[30]:


sns.histplot(x = "Fertilizer", data = df, kde = True)


# In[31]:


#Temperature Distribution


# In[32]:


sns.histplot(x="Temperatue", data = df, kde = True)


# In[33]:


#Macronutrients (NPK) Distribution


# In[34]:


fig, ax = plt.subplots(1,3,figsize=(10, 5))
sns.histplot(x = "Nitrogen (N)", data = df, kde = True, ax = ax[0])
sns.histplot(x = "Phosphorus (P)", data = df, kde = True, ax = ax[1])
sns.histplot(x = "Potassium (K)", data = df, kde = True, ax = ax[2])


# In[35]:


#Yield Distribution


# In[36]:


sns.histplot(x = "Yeild (Q/acre)", data = df, kde = True)


# In[37]:


#Rainfall and crop yield


# In[38]:


sns.scatterplot(x = 'Rain Fall (mm)', y = 'Yeild (Q/acre)', data = df)


# In[39]:


#Ferilizer and Crop Yield


# In[40]:


sns.scatterplot(x = 'Fertilizer', y = 'Yeild (Q/acre)', data = df)


# In[41]:


#Temperature and Crop Yield


# In[42]:


sns.scatterplot(x = 'Temperatue', y = 'Yeild (Q/acre)', data = df)


# In[43]:


#Macronutrients and Crop Yield


# In[44]:


fig, ax = plt.subplots(1,3,figsize=(15, 5))
sns.regplot(x = 'Nitrogen (N)', y = 'Yeild (Q/acre)', data = df, ax = ax[0])
sns.regplot(x = 'Phosphorus (P)', y = 'Yeild (Q/acre)', data = df, ax = ax[1])
sns.regplot(x = 'Potassium (K)', y = 'Yeild (Q/acre)', data = df, ax = ax[2])


# In[45]:


#Correlation Matrix Heatmap


# In[46]:


sns.heatmap(df.corr(), annot = True)


# In[47]:


#Train Test Split


# In[48]:


from sklearn.model_selection import train_test_split

# Define the target variable (y) and features (X)
X = df.drop('Yeild (Q/acre)', axis=1)  # Drop the target column from features
y = df['Yeild (Q/acre)']  # Target variable

# Split the dataset into training and testing sets (80% train, 20% test by default)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes to verify the split
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# In[ ]:




