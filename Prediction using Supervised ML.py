#!/usr/bin/env python
# coding: utf-8

# Author- Mehnaz khan
# 
# Task 1:- Preddiction Using Supervvised ML
# 

# In this task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied.

# In[ ]:


# Importing all necessary libraries
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Reading data
import pandas as pd
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)


# In[4]:


# Plotting the distribution of scores
import matplotlib.pyplot as plt
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

From the above graph, we can see that there present a positive linear relation betweeen the no. of hours studied and percentage of score.
# # Preparing the Data

# In[5]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values 


# The next step is to split this data into training and test sets.

# In[6]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)


# # Training of the Algorithm

# In[7]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[8]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# # Making the Predictions

# In[9]:


print(X_test) 
y_pred = regressor.predict(X_test) 


# In[10]:


# Comparing Actual value vs Predicted value
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# # EVALUATION

# In[16]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




