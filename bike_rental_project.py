
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np


# In[2]:


#Set the working directory

os.chdir("D:/Project_2")
os.getcwd()


# In[3]:


bike_data = pd.read_csv("day.csv", encoding = 'ISO - 8859 -1')


# In[4]:


bike_data.shape


# In[5]:


#Check for missing value
bike_data.isnull().sum()

#The data set does not contain any missing value


# In[ ]:


bike_data.info()


# # Convert the required data type 

# In[6]:


#Convert into category
for col in ['season','yr','mnth','holiday','weekday','workingday','weathersit']:
        bike_data[col] = bike_data[col].astype('object')
        


# In[7]:


#Convert into Numericals

for col in ['instant','temp','atemp','hum','windspeed','casual','registered','cnt']:
        bike_data[col] = bike_data[col].astype('float')         


# In[8]:


from datetime import datetime


# In[9]:


bike_data['dteday'].apply(str)


# In[10]:


#Convetr the Date variable

#bike_data['dteday'] = bike_data['dteday'].astype('Date')

bike_data['dteday'] = pd.to_datetime(bike_data['dteday'])


# # Check for missing values

# In[11]:


bike_data.isnull().sum()


# In[ ]:


#There is no missing value present in the given dataset


# # Outlier Analysis

# In[12]:


cnames = ["instant","temp","atemp","hum","windspeed","casual","registered","cnt"]


# In[13]:


#Select and Remove the outliers

for i in cnames:
    q75 , q25 = np.percentile(bike_data.loc[:,i],[75,25])
    iqr = q75  -  q25
    
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    
    print(min)
    print(max)
    
    bike_data = bike_data.drop(bike_data[bike_data.loc[:,i] < min].index)
    bike_data = bike_data.drop(bike_data[bike_data.loc[:,i] > max].index)


# In[14]:


print(min)


# In[15]:


print(max)


# In[16]:


bike_data.shape


# In[ ]:


#731-676

#55 observations got deleted


# # Feature Selection

# In[17]:


corr_plot = bike_data.loc[:,cnames]
corr_plot.info()


# In[18]:


import seaborn as sns


# In[19]:


get_ipython().magic('matplotlib inline')


# In[20]:


import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(7,5))

#Generate Correlation Matrix

corr = corr_plot.corr()

#plot using seaborn library


sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 50, as_cmap=True),square=True, ax=ax)


# In[21]:


#temp and atemp can be observed as they are highly correlated
#Thus we need to drop one of them
#Temp is the actual temperature whereas atemp is the feeling value of temperature
#Feeling temperature is more impprtant when it comes to real life


#Also casual, registered and cnt are positively correlated
#Moreover sum of casual and registered forms cnt variable


# In[22]:


#get variable names 
bike_data.columns.values


# In[23]:


bike_data = bike_data.drop(["instant","casual","registered","temp","dteday"],axis = 1)


# In[24]:


bike_data.columns.values


# In[25]:


#Till here, our features are selected and they are already normalized.
# We don't need to perform feature scaling
bike_data.to_csv("Clean_bike_data_python.csv",sep="\t")


# # Regression Model Development 

# In[ ]:


#__ Decision Tree Model ________


# In[27]:


from random import randrange, uniform


# In[28]:


import sklearn


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


bike_data = bike_data.drop(["dteday"],axis=1)


# In[36]:


#Divide the data into two part for training the model and testing the model

x = bike_data.values[:, 0:10]
y = bike_data.values[:,10]
y = y.astype('int')

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.2)


# In[33]:


from sklearn import tree


# In[39]:


import numpy as np


# In[37]:


c50_model = tree.DecisionTreeClassifier(criterion='entropy').fit(x_train, y_train)



# In[50]:


#predict new test cases
C50_Predictions = C50_model.predict(x_test)


# In[ ]:


#Create a definition of RMSLE error coefficient


# In[54]:


def rmsle(target, predicted):
    log1 = np.nan_to_num(np.array([np.log(v+1) for v in target]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in predicted]))
    calc = (log1 - log2) **2
    return np.sqrt(np.mean(calc))


# In[57]:


print("RMSLE Value :", rmsle(y_test,C50_Predictions))


# In[ ]:


##______ RANDOM FOREST MODEL _______


# In[42]:


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators = 20).fit(x_train, y_train)


# In[ ]:


#Predict values using Random Forest Model


# In[43]:


rf_predictions = rf_model.predict(x_test)


# In[58]:


#Check RMSLE error coefficient

print("RMSLE Value: ", rmsle(y_test,rf_predictions))

