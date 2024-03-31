#!/usr/bin/env python
# coding: utf-8

# # CREDIT CARD ATTRITION RATE

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import chi2_contingency
import numpy as np
import seaborn as sns


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('/Users/aparajith/Downloads/credit_card_churn.csv')


# In[ ]:


data.shape


# Removing the last 2 columns of NAIVE_BAYES_CLASSIFICATION, as suggested in the Problem Statement

# In[ ]:


data = data.drop(
['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],
axis = 1)


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.dtypes


# - 'int64' & 'float64' here shows the continuous variables
# - 'object' here shows the categorical variables

# #  UNIVARIATE ANALYSIS

# In[ ]:


data.describe()


# In[ ]:


data['Customer_Age'].plot.hist()


# - most of the customers lies between 45 - 50 years.

# In[ ]:


data['Attrition_Flag'].value_counts().plot.bar()


# - maximum is the ratio of the Existing Customer in the dataset.

# In[ ]:


data['Education_Level'].value_counts().plot.bar()


# - Highest Educational Qualification of maximum number of the customers is 'Graduate'.

# In[ ]:


data['Income_Category'].value_counts().plot.bar()


# - Maximum number of customers are from 'Less than $40k' income group annually.

# In[ ]:


data['Card_Category'].value_counts().plot.bar()


# - maximum number of customers have access to the 'Blue' card, whereas the least number of customers have 'Platinum' card.

# # BIVARIATE ANALYSIS

# In[ ]:


data['Customer_Age'].corr(data['Credit_Limit'])


# In[ ]:


data.plot.scatter('Customer_Age', 'Credit_Limit')


# - we can see that Customer of 40 - 50 age group has the maximum Credit Limit.

# In[ ]:


data['Customer_Age'].corr(data['Total_Trans_Amt'])


# In[ ]:


data.plot.scatter('Customer_Age', 'Total_Trans_Amt')


# - Total Transaction Amount between 1000 to 5000 is dense, transacted mostly by 37 - 57 age group people.

# In[ ]:


data.groupby('Attrition_Flag')['Customer_Age'].mean()


# - mean age of Attrited as well as Existing customers are almost same.

# In[ ]:


data.groupby('Gender')['Customer_Age'].mean()


# - mean age of Male as well as Female customers are almost same.

# In[ ]:


data.groupby('Card_Category')['Total_Relationship_Count'].mean().plot.bar()


# - we can see that as the Card Category is moving as "Blue > Silver > Gold > Platinum" the number of mean products held by the customers are decreasing.

# In[ ]:


data.groupby('Gender')['Credit_Limit'].mean().plot.bar()


# - Females have lower credit limit when compared to the males

# In[ ]:


data.groupby('Income_Category')['Credit_Limit'].mean().plot.bar()


# - As usual more income category customer('120K+ dollars') have highest credit limit & low income category customer('Less than 40K dollars') has lowest credit limit.

# In[ ]:


data.groupby('Card_Category')['Credit_Limit'].mean().plot.bar()


# - Card_Category in descending order i.e. "Platinum > Gold > Silver > Blue" has the Credit limit i.e. maximum credit limit for Platinum cardholders & least credit limit for Blue cardholders.

# In[ ]:


pd.crosstab(data['Gender'], data['Attrition_Flag'])


# In[ ]:


gen_bar = pd.crosstab(data['Gender'], data['Attrition_Flag'])
gen_bar.div(gen_bar.sum(axis = 1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (7,7))
plt.xlabel('Gender')
plt.ylabel('Percentage')


# - So as we can see from the graph that Female customers have a higher attrition rate than the Male customers.

# In[ ]:


pd.crosstab(data['Education_Level'], data['Attrition_Flag'])


# In[ ]:


mar_bar = pd.crosstab(data['Education_Level'], data['Attrition_Flag'])
mar_bar.div(mar_bar.sum(axis = 1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (8,8))
plt.xlabel('Education_Level')
plt.ylabel('Percentage')


# - Customers with 'Doctorate' followed by 'Post-Graduate' Educational Qualification rate have a higher attrition rate when compared to others.

# In[ ]:


data['Attrition_Flag'].replace('Existing Customer', 1, inplace = True)
data['Attrition_Flag'].replace('Attrited Customer', 0, inplace = True)


# - To check the correlation of our Target Variable('Attrition_Flag') we have converted their categorical value to the numerical values. As we can see correlation only between the numeric variables.

# In[ ]:


corr = data.corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots()
fig.set_size_inches(20, 10)
sns.heatmap(corr, mask = mask, vmax = .9, annot = True, cmap = 'inferno')


# - Among all other variables, 'Total_Trans_Ct' is higlhy coorelated with our target variable 'Attrition_Flag' which means that the more the transaction count, increased will be the customer activity resulting into Account still existing.

# # Missing Values & Outlier Treatment

# In[ ]:


data.isnull().sum()


# - So, we can see that there are no missing values in our data.

# In[ ]:


data['Customer_Age'].plot.box()


# In[ ]:


data.loc[data['Customer_Age'] > 68, 'Customer_Age'] = np.mean(data['Customer_Age'])


# - removing the outliers from 'Customer_Age', as there are some outliers above 68, so we will impute it with mean 'Customer_Age'.

# In[ ]:


data['Customer_Age'].plot.box()


# In[ ]:


data['Card_Category'].value_counts().plot.bar()


# - Here we can see that 'Platinum' & 'Gold' are the Outliers.

# In[ ]:


data['Card_Category'].replace('Gold', 'Silver', inplace = True)
data['Card_Category'].replace('Platinum', 'Silver', inplace = True)


# In[ ]:


data['Card_Category'].value_counts().plot.bar()


# - So we have imputed 'Gold' & 'Platinum' Card_Category with the 'Silver' Card_Category.

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# - this 'warnings' library will ignore the errors.

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# - Machine Learning algorithms can only work on numbers and not on labels, so we have to convert labels in these datasets into numbers using LABEL ENCODER.

# In[ ]:


le_Gender = LabelEncoder()
le_Education_Level = LabelEncoder()
le_Marital_Status = LabelEncoder()
le_Income_Category = LabelEncoder()
le_Card_Category = LabelEncoder()


# In[ ]:


data['Gender_n'] = le_Gender.fit_transform(data['Gender'])
data['Education_Level_n'] = le_Gender.fit_transform(data['Education_Level'])
data['Marital_Status_n'] = le_Gender.fit_transform(data['Marital_Status'])
data['Income_Category_n'] = le_Gender.fit_transform(data['Income_Category'])
data['Card_Category_n'] = le_Gender.fit_transform(data['Card_Category'])


# In[ ]:


data.head()


# In[ ]:


data_n = data.drop(['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'], axis = 1)


# In[ ]:


data_n.head()


# In[ ]:


data_n = data_n.drop('CLIENTNUM', axis = 1)


# In[ ]:


data_n.shape


# - we have removed the 'CLIENTNUM' column before preparing our model, because Client Number is not useful for predicting our model.
# - As we have done all the exploratory analysis, now it's time to build our model to predict the Customer Attrition.

# # Model Building

# In[ ]:


train = data_n.drop('Attrition_Flag',  axis = 1)
target = data_n['Attrition_Flag']


# - 'train' contains our Independent Variables.
# - 'target' contains our Target Variable.
# - now we will split our training and testing data into 81 : 19.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(train, target, test_size = 0.19, random_state = 20)


# - as our data is ready now, we will built Logistic Regression model, as our target variable is Discrete in nature.

# LOGISTIC REGRESSION

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logreg = LogisticRegression()


# In[ ]:


logreg.fit(x_train, y_train)


# - fitting our training data into the model.

# In[ ]:


prediction = logreg.predict(x_test)


# - doing 'prediction' on the testing dataset.
# - now we will evaluate, that how accurate our model  is, by computing the 'accuracy score' of the test dataset.

# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test, prediction)


# - we got an accuracy of 90% on our test dataset. Logistic Regression has a Linear Decision Boundary.
# - What if our data have non - linearity?
# - So, we need a model which can capture this non - linearity.
# - So, now we will try to fit our data on Decision Tree algorithm, to check if we can get better accuracy with it.

# DECISION TREE

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


clf = DecisionTreeClassifier(random_state = 20)


# In[ ]:


clf.fit(x_train, y_train)


# - fitting our training data into the model.

# In[ ]:


prediction_clf = clf.predict(x_test)


# - doing 'prediction' on the testing dataset.
# - now we will evaluate how accurate our model is, by computing the 'accuracy score' of the test dataset.

# In[ ]:


accuracy_score(y_test, prediction_clf)


# - So, we got an accuracy of 93% i.e. more than accuracy of the Logistic Regression model.
