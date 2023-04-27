#!/usr/bin/env python
# coding: utf-8

# # Contraceptive Method Choice of Indonesian Woman
# 
# ## Introduction
# 
# ### Authored by:
# #### Team Name : ELITE
# 
# Team Members:Sindhura Alla,Medha Alla,Ravindra Kumar Velidandi,Sai Mithil Sagi,Venkata Saipavan Lahar Sudrosh Kumar Atchutha,Sanjana Thinderu,
# 
# ### Description of the analysis
# In this project, we are using a dataset containing 1987 National Indonesia Contraceptive Prevalence Survey from UCI's Machine Learning Repository:https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice
# 
# Our prediction task is to determine the current contraceptive method choice (no use, long-term methods, or short-term methods) of a woman based on her demographic and socio-economic characteristics.
# 
#  We are using the input variables that include wife age,wife education,Husband Education,children,wife religion,wife woking,husband occupation,standard of living,media exposure.
#  
#  Attribute Information:
# 
# 1. Wife's age (numerical)
# 2. Wife's education (categorical) 1=low, 2, 3, 4=high
# 3. Husband's education (categorical) 1=low, 2, 3, 4=high
# 4. Number of children ever born (numerical)
# 5. Wife's religion (binary) 0=Non-Islam, 1=Islam
# 6. Wife's now working? (binary) 0=Yes, 1=No
# 7. Husband's occupation (categorical) 1, 2, 3, 4
# 8. Standard-of-living index (categorical) 1=low, 2, 3, 4=high
# 9. Media exposure (binary) 0=Good, 1=Not good
# 10. Contraceptive method used (class attribute) 1=No-use, 2=Long-term, 3=Short-term
# 
# 
#  
# Objective is to tune the given models:
# K-NN
# Decision Tree 
# Random Forest
# AdaBoost
# GradientDescent
# XGBoost
# Evaluating the performance of these models, and selecting which is the best model for the contraceptive dataset.  
# 
# 
# 

# ## Step 1 - Importing the required packages

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier


# ## Step 2 - Preliminary (Business) Problem Scoping
# 
# We are developing a multi-class classifier  to determine the current contraceptive method choice (no use, long-term methods, or short-term methods) of a woman based on her demographic and socio-economic characteristics.. It is not yet known of the classes are imbalanced. If these classes are imbalanced, we will look to rebalance them using a oversampling techniqe. 

# ## Step 3 - Loading, cleaning and preparing data

# ### Loading the data from data source

# In[2]:


contra_df=pd.read_csv("contraceptive data.csv")
contra_df.head(5)


# ### Data Exploration

# In[3]:


contra_df.columns


# In[4]:


contra_df.describe()


# In[5]:


contra_df.info()


# #### Our findings from the data exploration indicate the the data requires a renaming/cleanup of column names. We also note that wife education  ,Husband Education,husband occupation,standard of living are categorical data but still have the numerical values. 

# ## Step 4 - Cleaning/transforming data (where necessary)

# #### cleaning the column names

# In[6]:


contra_df.columns = [col.strip().lower().replace(' ', '_') for col in contra_df.columns]
contra_df.columns


# In[7]:


contra_df['contraceptive_used'].unique()


# ### Checking for null values
# 

# In[8]:


contra_df.isnull().sum()


# In[9]:


### Categorizing required column
contra_df['wife_education']=contra_df['wife_education'].astype('category')
contra_df['husband_education']=contra_df['husband_education'].astype('category')
contra_df['standard_of_living']=contra_df['standard_of_living'].astype('category')
contra_df['husband_occupation']=contra_df['husband_occupation'].astype('category')
contra_df['contraceptive_used']=contra_df['contraceptive_used'].astype('category')
contra_df['contraceptive_used']=contra_df['contraceptive_used'].astype('category')
contra_df['media_exposure']=contra_df['media_exposure'].astype('category')
contra_df['wife_woking']=contra_df['wife_woking'].astype('category')
contra_df=contra_df.drop(columns=['wife_religion'])


# In[10]:


contra_df.dtypes


# In[11]:


contra_df['contraceptive_used'].value_counts()


# ### we can see that the data is slightly imbalanced so we use over sampling technique for KNN as it is sensitive to it

# ## Step 5 - Split data into training and validation sets

# #### Create the training set and the test set with a 70/30 split.
# We've decided to utilize a training/test split of the data at 70% training and 30% testing. This percentage split ratio is inline with common practice for small to medium sized datasets, which this data represents. Moreover, we have decided not to do a three way data split, as we are only testing two models and we wish to allocated as much data as possible to training and validation steps.

# In[12]:


train_df,test_df=train_test_split(contra_df, test_size=0.3, random_state=11)


# In[13]:


train_df


# In[14]:


train_X = train_df.drop(columns=['contraceptive_used'])
train_Y = train_df.contraceptive_used                      
test_X = test_df.drop(columns=['contraceptive_used'])
test_Y = test_df.contraceptive_used


# In[15]:


train_X


# # Train our models

# ## 7.1 Training a pruned decision tree using RandomixedserachCv ad GridsearchCv:

# Determine the parameters that can be "tuned"
# 
# You can review the parameters of the model which you're trying to "tune". In this case, we're using a DecisionTreeClassifier. Begin by reviewing the parameters for this model found [here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
# fter reviewing these parameters (while also understanding something about DecisionTrees), we can identify the following parameters that could affect model fit. 
# 
# * criterion
# * max_depth
# * min_samples_split
# * min_samples_leaf
# * max_leaf_nodes
# * min_impurity_decrease
# 
# 

# ### Create an initial 'wide' range of possible hyperparameter values

# In[16]:



criterion = ['gini', 'entropy']
max_depth = [int(x) for x in np.linspace(1, 500, 50)]
min_samples_split = [int(x) for x in np.linspace(2, 500, 50)]
min_samples_leaf = [int(x) for x in np.linspace(1, 100, 50)]
max_leaf_nodes = [int(x) for x in np.linspace(2, len(test_Y), 50)]
min_impurity_decrease = [x for x in np.arange(0.0, 0.01, 0.0001).round(5)]
param_grid_random = { 'criterion': criterion,
                      'max_depth': max_depth,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf' : min_samples_leaf,
                      'max_leaf_nodes' : max_leaf_nodes,
                      'min_impurity_decrease' : min_impurity_decrease,
                     }


# ### Using Randomize Search to narrow the possible range of parameter values

# In[17]:


random_seed=11
dtree_default = DecisionTreeClassifier(random_state=random_seed)
# change n_iter to 200_000 for full run
best_random_search_model = RandomizedSearchCV(
        estimator=DecisionTreeClassifier(random_state=random_seed), 
        scoring='recall', 
        param_distributions=param_grid_random, 
        n_iter = 5_000, 
        cv=10, 
        verbose=0, 
        n_jobs = -1,
        random_state=random_seed
    )
best_random_search_dtree_model = best_random_search_model.fit(train_X, train_Y)


# In[18]:


random_search_best_dtree_params = best_random_search_dtree_model.best_params_
print('Best parameters found: ', random_search_best_dtree_params)


# ### Test the performance of the selected parameters

# In[19]:


y_pred = best_random_search_dtree_model.predict(test_X)

print("************************************")
print(f"{'Recall Score:':18}{recall_score(test_Y, y_pred,average='weighted')}")
print("************************************")
print(f"{'Accuracy Score: ':18}{accuracy_score(test_Y, y_pred)}")
print(f"{'Precision Score: ':18}{precision_score(test_Y, y_pred,average='weighted')}")
print(f"{'F1 Score: ':18}{f1_score(test_Y, y_pred,average='weighted')}")
print("************************************")


# In[20]:


confusion_matrix(test_Y, y_pred)


# In[21]:


random_search_best_dtree_params


# ### GRID SEARCH

# In[22]:


plus_minus = 8 # change this to 10-15 when doing a final run. this current value is for testing
increment = 2

param_grid = { 'min_samples_split': [x for x in range(random_search_best_dtree_params['min_samples_split']-plus_minus, random_search_best_dtree_params['min_samples_split']+plus_minus,2) if x >= 2],       
              'min_samples_leaf': [x for x in range(random_search_best_dtree_params['min_samples_leaf']-plus_minus , random_search_best_dtree_params['min_samples_leaf']+plus_minus,2) if x > 0],
              'min_impurity_decrease': [x for x in np.arange(random_search_best_dtree_params['min_impurity_decrease']-0.001, random_search_best_dtree_params['min_impurity_decrease']+0.001,.0001).round(5) if x >= 0.000],
              'max_leaf_nodes':[x for x in range(random_search_best_dtree_params['max_leaf_nodes']-plus_minus , random_search_best_dtree_params['max_leaf_nodes']+plus_minus, 2) if x > 1],  
              'max_depth': [x for x in range(random_search_best_dtree_params['max_depth']-plus_minus , random_search_best_dtree_params['max_depth']+plus_minus, 2) if x > 1],
              'criterion': [random_search_best_dtree_params['criterion']]
              }
best_grid_search_model = GridSearchCV(estimator=DecisionTreeClassifier(random_state=random_seed), 
                                    scoring='recall', param_grid=param_grid, cv=10, verbose=0,  n_jobs = -1)
best_grid_search_dtree_model = best_grid_search_model.fit(train_X, train_Y)


# In[23]:


print('Best parameters found: ', best_grid_search_dtree_model.best_params_)


# In[24]:


y_pred = best_grid_search_dtree_model.predict(test_X)
print("************************************")
print(f"{'Recall Score:':18}{recall_score(test_Y, y_pred,average='weighted')}")
print("************************************")
print(f"{'Accuracy Score: ':18}{accuracy_score(test_Y, y_pred)}")
print(f"{'Precision Score: ':18}{precision_score(test_Y, y_pred,average='weighted')}")
print(f"{'F1 Score: ':18}{f1_score(test_Y, y_pred,average='weighted')}")
print("************************************")


# ## 7.1 Training a RandomForest using RandomixedserachCv ad GridsearchCv:

# Like all our classifiers, RandomeForestClassifier has a number of parameters that can be adjusted/tuned. In this example below, we simply accept the defaults. You may want to experiment with changing the defaul values and also use GridSearchCV to explore ranges of values.
# 
# * n_estimators: The number of trees in the forsest
#     - A deeper tree might increase the performance, but also the complexity and chances to overfit.
#     - The value must be an integer greater than 0. Default is 100.  
# * max_depth: The maximum depth per tree. 
#     - Deeper trees might increase the performance, but also the complexity and chances to overfit.
#     - The value must be an integer greater than 0. Default is None, which allows the tree to grow without constraint.
# * criterion
# * min_samples_split
# * min_samples_leaf
# * max_leaf_nodes
# * min_impurity_decrease

# ### Create an initial 'wide' range of possible hyperparameter values

# In[25]:


criterion = ['gini', 'entropy']
max_depth = [int(x) for x in np.linspace(1, 500, 50)]
min_samples_split = [int(x) for x in np.linspace(2, 500, 50)]
min_samples_leaf = [int(x) for x in np.linspace(1, 100, 50)]
max_leaf_nodes = [int(x) for x in np.linspace(2, len(test_Y), 50)]
min_impurity_decrease = [x for x in np.arange(0.0, 0.01, 0.0001).round(5)]
param_grid_random = { 
                      'criterion': criterion,
                      'max_depth': max_depth,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf' : min_samples_leaf,
                      'max_leaf_nodes' : max_leaf_nodes,
                      'min_impurity_decrease' : min_impurity_decrease,
                     }


# ### Using Randomize Search to narrow the possible range of parameter values

# In[26]:


random_seed=11
randomtree_default = RandomForestClassifier(random_state=random_seed)
# change n_iter to 200_000 for full run
best_random_search_model = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=random_seed), 
        scoring='recall', 
        param_distributions=param_grid_random, 
        n_iter = 5_000, 
        cv=10, 
        verbose=0, 
        n_jobs = -1,
        random_state=random_seed
    )
best_random_search_rtree_model = best_random_search_model.fit(train_X, train_Y)


# In[27]:


random_search_best_rtree_params = best_random_search_model.best_params_
print('Best parameters found: ', random_search_best_rtree_params)


# In[28]:


y_pred = best_random_search_rtree_model.predict(test_X)

print("************************************")
print(f"{'Recall Score:':18}{recall_score(test_Y, y_pred,average='weighted')}")
print("************************************")
print(f"{'Accuracy Score: ':18}{accuracy_score(test_Y, y_pred)}")
print(f"{'Precision Score: ':18}{precision_score(test_Y, y_pred,average='weighted')}")
print(f"{'F1 Score: ':18}{f1_score(test_Y, y_pred,average='weighted')}")
print("************************************")


# In[29]:


random_search_best_rtree_params


# ### GRID SEARCH

# In[ ]:


plus_minus = 10 # change this to 10-15 when doing a final run. this current value is for testing
increment = 2

param_grid = { 'min_samples_split': [x for x in range(random_search_best_rtree_params['min_samples_split']-plus_minus, random_search_best_rtree_params['min_samples_split']+plus_minus,2) if x >= 2],       
              'min_samples_leaf': [x for x in range(random_search_best_rtree_params['min_samples_leaf']-plus_minus , random_search_best_rtree_params['min_samples_leaf']+plus_minus,2) if x > 0],
              'min_impurity_decrease': [x for x in np.arange(random_search_best_rtree_params['min_impurity_decrease']-0.001, random_search_best_rtree_params['min_impurity_decrease']+0.001,.0001).round(5) if x >= 0.000],
              'max_leaf_nodes':[x for x in range(random_search_best_rtree_params['max_leaf_nodes']-plus_minus , random_search_best_rtree_params['max_leaf_nodes']+plus_minus, 2) if x > 1],  
              'max_depth': [x for x in range(random_search_best_rtree_params['max_depth']-plus_minus , random_search_best_rtree_params['max_depth']+plus_minus, 2) if x > 1],
              'criterion': [random_search_best_rtree_params['criterion']]
              }
best_grid_search_model = GridSearchCV(estimator=RandomForestClassifier(random_state=random_seed), 
                                    scoring='recall', param_grid=param_grid, cv=10, verbose=0,  n_jobs = -1)
best_grid_search_rtree_model = best_grid_search_model.fit(train_X, train_Y)


# In[ ]:


print('Best parameters found: ', best_grid_search_rtree_model.best_params_)


# In[ ]:


y_pred = best_grid_search_rtree_model.predict(test_X)
print("************************************")
print(f"{'Recall Score:':18}{recall_score(test_Y, y_pred,average='weighted')}")
print("************************************")
print(f"{'Accuracy Score: ':18}{accuracy_score(test_Y, y_pred)}")
print(f"{'Precision Score: ':18}{precision_score(test_Y, y_pred,average='weighted')}")
print(f"{'F1 Score: ':18}{f1_score(test_Y, y_pred,average='weighted')}")
print("************************************")


# ## Training  AdaBoost using RandomixedserachCv ad GridsearchCv:

# * max_depth: The maximum depth per tree. 
#     - A deeper tree might increase the performance, but also the complexity and chances to overfit.
#     - The value must be an integer greater than 0. Default is None (meaning, the tree can grow to a point where all leaves have 1 observation).
# * learning_rate: The learning rate determines the step size at each iteration while your model optimizes toward its objective. 
#     - A low learning rate makes computation slower, and requires more rounds to achieve the same reduction in residual error as a model with a high learning rate. But it optimizes the chances to reach the best optimum.
#     - Larger learning rates may not converge on a solution.
#     - The value must be between 0 and 1. Default is 0.3.
# * n_estimators: The number of trees in our ensemble. 
#     - Equivalent to the number of boosting rounds.
#     - The value must be an integer greater than 0. Default is 100.

# ### Create an initial 'wide' range of possible hyperparameter values

# In[ ]:


n_estimators = [int(x) for x in np.linspace(1, 100, 5)]
learning_rate = [float(x) for x in np.linspace(1, 10, 1)]
param_grid_random = { 'n_estimators': n_estimators,
                      'learning_rate': learning_rate,
                     }


# ### Using Randomize Search to narrow the possible range of parameter values
# 

# In[ ]:


random_seed=11
best_random_search_model = RandomizedSearchCV(
        estimator=AdaBoostClassifier(random_state=random_seed), 
        scoring='recall', 
        param_distributions=param_grid_random, 
        n_iter = 5_000, 
        cv=10, 
        verbose=0, 
        n_jobs = -1,
        random_state=random_seed
    )
best_random_search_Ada_model = best_random_search_model.fit(train_X, train_Y)


# In[ ]:


random_search_best_Ada_params = best_random_search_Ada_model.best_params_
print('Best parameters found: ', random_search_best_Ada_params)


# In[ ]:


y_pred = best_random_search_Ada_model.predict(test_X)

print("************************************")
print(f"{'Recall Score:':18}{recall_score(test_Y, y_pred,average='weighted')}")
print("************************************")
print(f"{'Accuracy Score: ':18}{accuracy_score(test_Y, y_pred)}")
print(f"{'Precision Score: ':18}{precision_score(test_Y, y_pred,average='weighted')}")
print(f"{'F1 Score: ':18}{f1_score(test_Y, y_pred,average='weighted')}")
print("************************************")


# In[ ]:


random_search_best_Ada_params


# ### GRID SEARCH

# In[ ]:


lus_minus = 10 # change this to 10-15 when doing a final run. this current value is for testing
increment = 2
param_grid = { 'n_estimators': [x for x in range(random_search_best_Ada_params['n_estimators']-plus_minus, random_search_best_Ada_params['n_estimators']+plus_minus,2) if x >= 2],       
              'learning_rate': [x for x in range(random_search_best_Ada_params['learning_rate']-plus_minus , random_search_best_Ada_params['learning_rate']+plus_minus,2) if x > 0],
              
best_grid_search_model = GridSearchCV(estimator=AdaBoostClassifier(random_state=random_seed), 
                                    scoring='recall', param_grid=param_grid, cv=10, verbose=0,  n_jobs = -1)
best_grid_search_Ada_model = best_grid_search_model.fit(train_X, train_Y)


# In[ ]:


print('Best parameters found: ', best_grid_search_Ada_model.best_params_)


# In[ ]:


y_pred = best_grid_search_Ada_model.predict(test_X)
print("************************************")
print(f"{'Recall Score:':18}{recall_score(test_Y, y_pred,average='weighted')}")
print("************************************")
print(f"{'Accuracy Score: ':18}{accuracy_score(test_Y, y_pred)}")
print(f"{'Precision Score: ':18}{precision_score(test_Y, y_pred,average='weighted')}")
print(f"{'F1 Score: ':18}{f1_score(test_Y, y_pred,average='weighted')}")
print("************************************")


# ##  Training  GradientBoostingClassifier using RandomixedserachCv ad GridsearchCv:

# * max_depth: The maximum depth per tree. 
#     - A deeper tree might increase the performance, but also the complexity and chances to overfit.
#     - The value must be an integer greater than 0. Default is None (meaning, the tree can grow to a point where all leaves have 1 observation).
# * learning_rate: The learning rate determines the step size at each iteration while your model optimizes toward its objective. 
#     - A low learning rate makes computation slower, and requires more rounds to achieve the same reduction in residual error as a model with a high learning rate. But it optimizes the chances to reach the best optimum.
#     - Larger learning rates may not converge on a solution.
#     - The value must be between 0 and 1. Default is 0.3.
# * n_estimators: The number of trees in our ensemble. 
#     - Equivalent to the number of boosting rounds.
#     - The value must be an integer greater than 0. Default is 100.

# ### Create an initial 'wide' range of possible hyperparameter values

# In[ ]:


n_estimators = [int(x) for x in np.linspace(1, 100, 5)]
learning_rate = [float(x) for x in np.linspace(1, 10, 1)]
max_depth = [int(x) for x in np.linspace(1, 500, 50)]
param_grid_random = { 'n_estimators': n_estimators,
                      'learning_rate': learning_rate,
                      'max_depth':max_depth
                     }


# ### Using Randomize Search to narrow the possible range of parameter values
# 

# In[ ]:


random_seed=11
best_random_search_model = RandomizedSearchCV(
        estimator=GradientBoostingClassifier(random_state=random_seed), 
        scoring='recall', 
        param_distributions=param_grid_random, 
        n_iter = 5_000, 
        cv=10, 
        verbose=0, 
        n_jobs = -1,
        random_state=random_seed
    )
best_random_search_gb_model = best_random_search_model.fit(train_X, train_Y)


# In[ ]:


random_search_best_gb_params = best_random_search_gb_model.best_params_
print('Best parameters found: ', random_search_best_gb_params)


# In[ ]:


y_pred = best_random_search_gb_model.predict(test_X)

print("************************************")
print(f"{'Recall Score:':18}{recall_score(test_Y, y_pred,average='weighted')}")
print("************************************")
print(f"{'Accuracy Score: ':18}{accuracy_score(test_Y, y_pred)}")
print(f"{'Precision Score: ':18}{precision_score(test_Y, y_pred,average='weighted')}")
print(f"{'F1 Score: ':18}{f1_score(test_Y, y_pred,average='weighted')}")
print("************************************")


# In[ ]:


random_search_best_gb_params


# ### GRID SEARCH

# In[ ]:


lus_minus = 10 # change this to 10-15 when doing a final run. this current value is for testing
increment = 2

param_grid = { 'n_estimators': [x for x in range(random_search_best_gb_params['n_estimators']-plus_minus, random_search_best_gb_params['n_estimators']+plus_minus,2) if x >= 2],       
              'learning_rate': [x for x in range(random_search_best_gb_params['learning_rate']-plus_minus , random_search_best_gb_params['learning_rate']+plus_minus,2) if x > 0],
              'max_depth'    : [x for x in range(random_search_best_gb_params['max_depth']-plus_minus , random_search_best_gb_params['max_depth']+plus_minus, 2) if x > 1],
             }
best_grid_search_model = GridSearchCV(estimator=GradientBoostingClassifier(random_state=random_seed), 
                                    scoring='recall', param_grid=param_grid, cv=10, verbose=0,  n_jobs = -1)
best_grid_search_gb_model = best_grid_search_model.fit(train_X, train_Y)


# In[ ]:


print('Best parameters found: ', best_grid_search_gb_model.best_params_)


# In[ ]:


y_pred = best_grid_search_gb_model.predict(test_X)
print("************************************")
print(f"{'Recall Score:':18}{recall_score(test_Y, y_pred,average='weighted')}")
print("************************************")
print(f"{'Accuracy Score: ':18}{accuracy_score(test_Y, y_pred)}")
print(f"{'Precision Score: ':18}{precision_score(test_Y, y_pred,average='weighted')}")
print(f"{'F1 Score: ':18}{f1_score(test_Y, y_pred,average='weighted')}")
print("************************************")

