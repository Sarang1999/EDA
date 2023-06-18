#!/usr/bin/env python
# coding: utf-8

# # Task3 - Exploratory Data analysis - Retail
# Present by- Sarang Sonawane
Perform Exploratory Data Analysis on dataset "SampleSuperstore"
Tools:- Python, Numpy, Pandas, Matplotlib, Seaborn 
Problem Statement:- As a business manager, Try to find out The weak areas where you can work to make more profit 
# In[1]:


# import necessary libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# read csv file 
df = df = pd.read_csv('C:\\Users\\HP\\Downloads\\SampleSuperstore.csv')


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


# the shape of dataset is 9994 rows and 13 column we find  
df.shape


# In[7]:


# column name is there 
df.columns


# In[8]:


# the imformation about the dataset how many object is there, how many int,float value is there.
df.info()


# In[9]:


# find null values in datasets for using isnull() function
df.isnull()


# In[10]:


# the sum of  null values in the datasets
df.isnull().sum()


# In[11]:


df.describe()


# In[12]:


# find duplicate value in datasets
df.duplicated().sum()


# In[13]:


# in the dataset, if duplicate value are available else you have to delete this value by using drop duplicate function 
# here you can see i am deleting duplicates in our dataset
df.drop_duplicates(inplace = True)


# In[14]:


df.duplicated().sum()


# In[15]:


# find the unique values and string values in the datasets by using nunique function 
df.nunique()


# In[16]:


#by using unique function, you can find out the unique category in column 
# here you can see the unique shipe mode in df.
print(df['Ship Mode'].unique())


# In[17]:


# similarly you can see the unique category in Category column  
print(df['Category'].unique())
print("there are %d category in this df."%df['Category'].nunique())


# In[18]:


#then this is the unique States name in this set 
print(df['State'].unique())


# In[19]:


#similerly, you can see the number of unique entries in the column by using nunique() method
no_of_states = df['State'].nunique()
print("There are %d states in this df."%no_of_states)


# In[20]:


print(df['Sub-Category'].unique())


# In[21]:


no_of_subcategory=df['Sub-Category'].nunique()
print("Categories are divided into %d subcategories"%no_of_subcategory)


# In[22]:


df['Segment'].value_counts()


# In[23]:


df.describe()


# In[24]:


# creating loss of dataframe


# In[25]:


# lets create a new dataframe where profit is negative it means loss,its helps to improve weak areas.
# here i am taking new variable and assing the dataset and then aplying the condition,profit is less than 0
loss = df[df['Profit'] < 0]


# In[26]:


loss


# In[27]:


#view the shape 
loss.shape


# In[28]:


loss.describe()


# In[29]:


total_loss = np.negative(loss['Profit'].sum())


# In[30]:


print("total_loss = %.2d"% total_loss)


# In[31]:


#then by using groupby function, you can split the data into groups.its helps to analyze more easy 
loss.groupby('Segment').sum()

here you can see the output,the discount is high, i thing more discount leads to more loss
So to make profit, provide fewer discount
# In[32]:


loss.groupby('Sub-Category').sum()

here we can observe more loss in the binders, machine and table category when compared to other categories
# In[33]:


loss['Sub-Category'].value_counts()


# In[34]:


#highly loss of top 10 cities
loss.groupby('City').sum().sort_values('Profit',ascending=True).head(10)


# In[35]:


# we calculating an average and we observed that more loss in technology Category
loss.sort_values(['Sales'],ascending = True).groupby(['Category']).mean()


# In[36]:


# Calculating sales in state vise
# we are using the _nsmallest_ function to identifying less sales
# these are 10 states where sales are very less
df.groupby(['State']).sum()['Sales'].nsmallest(10)


# In[37]:


#Here consumer segment sales is less than other segment.
#But this segment provides higher profits compared to other segment.
#So, have to focus on sales in this segment by advertisement or something else then,
#for sure,we can gain more profit
df.sort_values(['Sales'],ascending=True).groupby('Segment').sum()


# In[38]:


#Here we can see that sales are less in the south region.So to get more profit or more sale
#we should fous on this area too.
df.groupby(['Region']).sum()


# In[39]:


# Data visuliazation


# In[40]:


import matplotlib.pyplot as plt


# In[41]:


# here we can see that the sales of fasteners,appliances, furnishings,accessories is very low.
plt.rcParams['figure.figsize']=(15,3)
plt.bar(loss['Sub-Category'],loss['Sales'],)
plt.rcParams.update({'font.size':10})
plt.xlabel('Sub_Category');
plt.ylabel('Sales');


# In[42]:


# When it comes to comparison in overall supermarket data, Fasteners, Labels, Furnishings, Art, paper, Envelopes, etc., 
#sub-categories have very fewer sales, that’s why it needs to be improved.
plt.rcParams['figure.figsize']=(28,6)
plt.bar(df['Sub-Category'],df['Sales'],)
plt.rcParams.update({'font.size':16});
plt.xlabel('Sub_Category');
plt.ylabel('Sales');


# In[43]:


plt.rcParams['figure.figsize']=(28,8)
plt.bar(df['Sub-Category'],df['Discount'],)
plt.rcParams.update({'font.size':14});
plt.xlabel('Sub_Category');
plt.ylabel('Discount');


# In[55]:


plt.rcParams['figure.figsize']=(15,4)
plt.bar(loss['Ship Mode'],loss['Sales'],)
plt.rcParams.update({'font.size':14})
plt.xlabel('Ship Mode');
plt.ylabel('Sales');
plt.show()

here we observed the sales are high if the ship mode is standard class and sales are low if the ship mode either second class or same day. 
# In[45]:


# import seaborn library for visualization


# In[46]:


import seaborn as sns


# In[56]:


plt.rcParams['figure.figsize']=(15,5)
sns.countplot(x=df.Segment)
plt.show();

here we see home office count is less. so improvement is important for this segment 
# In[48]:


plt.rcParams['figure.figsize']=(20,5)
plt.rcParams.update({'font.size':12})
sns.countplot(x='Category',data=df)
plt.show()

here we see the technology count is less. so we have to focus on this category for improvement in sales .
# In[49]:


# in the below data, its very much clear that the Copiers and Machine subcategory needs improvement
plt.rcParams['figure.figsize']=(20,5)
plt.rcParams.update({'font.size':12})
sns.countplot(x='Sub-Category',data=df)
plt.show()

Fron the above data, its very much clear that the Copiers and Machine subcategory needs improvement
# In[57]:


plt.rcParams['figure.figsize']=(20,5)
plt.rcParams.update({'font.size':12})
sns.countplot(x='Region',data=df)
plt.show()

Here we observed the south region needs more improvents campered to other region.Overoll analysis 
# if we have gain more profit then give a less discount to the customer 
# its better to give more discount in festival season 
# home office segment needs better improvements
# we have to focus thos cities where sales is less. we have to also focus on our product advertisement.
# so,its helps to increase our sales and profit.