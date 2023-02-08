#!/usr/bin/env python
# coding: utf-8

# ### Ensemble Learning

# In[12]:


import pandas as pd
import numpy as np
#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import MinMaxScaler


# In[13]:


# load the dataset in a DataFrame object
data = pd.read_csv('cancer.txt')
data.head()


# In[3]:


#"Sample code number" is just an indicator and IS of no use in the modeling. So, let's drop it.
data.drop(['Sample Code Number'],axis = 1, inplace = True)
data.head()


# In[4]:


#Get some statistics about the data 
data.describe()


# In[5]:


data.info()


# In[6]:


# The dataset contains missing values.We will replace those "?"s with 0's
data.replace('?',0, inplace=True)


# In[7]:


#The "?"s are replaced with 0's now. Let's do the missing value treatment now.
# Convert the DataFrame object into NumPy array otherwise you will not be able to impute
values = data.values

# Now impute it
#imputer = SimpleImputerImputer()
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputedData = imputer.fit_transform(values)


# In[8]:


#Now if you take a look at the dataset , we will see that all the ranges of the features of the dataset are not the same.
#This may cause a problem. A small change in a feature might not affect the other.
#To address this problem, you will normalize the ranges of the features to a uniform range, in this case, 0 - 1.
scaler = MinMaxScaler(feature_range=(0, 1))
normalizedData = scaler.fit_transform(imputedData)


# In[9]:


# Bagged Decision Trees for Classification - necessary dependencies

from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Segregate the features from the labels
X = normalizedData[:,0:9]
Y = normalizedData[:,9]
#print(Y)
kfold = model_selection.KFold(n_splits=10, random_state=7)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=7)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# The fundamental difference between bagging and random forest is that in Random forests, only a subset of features are selected at random out of the total and the best split feature from the subset is used to split each node in a tree, unlike in bagging where all features are considered for splitting a node.
# 
# The fundamental difference is that in Random forests, only a subset of features are selected at random out of the total and the best split feature from the subset is used to split each node in a tree, unlike in bagging where all features are considered for splitting a node.

# In[10]:


# AdaBoost Classification

from sklearn.ensemble import AdaBoostClassifier
seed = 7
num_trees = 70
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# In[11]:


# Voting Ensemble for Classification

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)

# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))

# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())


# In[ ]:




