#!/usr/bin/env python
# coding: utf-8

# ## Data loading

# In[1]:


import pandas as pd

control_data = pd.read_csv('data/control_data.csv')
experiment_data = pd.read_csv('data/experiment_data.csv')


# In[2]:


control_data.head()


# In[3]:


experiment_data.head()


# ## Number summaries and basic investigations

# In[4]:


control_data.info()


# In[5]:


experiment_data.info()


# In[6]:


control_data.isna().sum()


# In[7]:


experiment_data.isna().sum()


# In[8]:


control_data[control_data['Enrollments'].isna()]


# ## Data wrangling

# In[2]:


# Combine with Experiment data
data_total = pd.concat([control_data, experiment_data])
data_total.sample(10)


# In[3]:


import numpy as np
np.random.seed(7)
import sklearn.utils

# Add row id
data_total['row_id'] = data_total.index

# Create a Day of Week feature
data_total['DOW'] = data_total['Date'].str.slice(start=0, stop=3)

# Remove missing data
data_total.dropna(inplace=True)

# Add a binary column Experiment to denote
# if the data was part of the experiment or not (Random)
data_total['Experiment'] = np.random.randint(2, size=len(data_total))

# Remove missing data
data_total.dropna(inplace=True)

# Remove Date and Payments columns
del data_total['Date'], data_total['Payments']

# Shuffle the data
data_total = sklearn.utils.shuffle(data_total)


# In[11]:


# Check the new data
data_total.head()


# In[4]:


# Reorder the columns 
data_total = data_total[['row_id', 'Experiment', 'Pageviews', 'Clicks', 'DOW', 'Enrollments']]


# In[5]:


# Splitting the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_total.loc[:, data_total.columns != 'Enrollments'],                                                    data_total['Enrollments'], test_size=0.2)


# In[6]:


# Converting strings to numbers
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
X_train['DOW'] = lb.fit_transform(X_train['DOW'])
X_test['DOW'] = lb.transform(X_test['DOW'])


# In[17]:


X_train.head()


# In[18]:


X_test.head()


# ## Helper functions
# - Function for printing the evaluation scores related to a _regression_ problem
# - Function for plotting the original values and values predicted by the model

# In[7]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def calculate_metrics(y_test, y_preds):
    rmse = np.sqrt(mean_squared_error(y_test, y_preds))
    r_sq = r2_score(y_test, y_preds)
    mae = mean_absolute_error(y_test, y_preds)

    print('RMSE Score: {}'.format(rmse))
    print('R2_Squared: {}'.format(r_sq))
    print('MAE Score: {}'.format(mae))


# In[8]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

def plot_preds(y_test, y_preds, model_name):
    N = len(y_test)
    plt.figure(figsize=(10,5))
    original = plt.scatter(np.arange(1, N+1), y_test, c='blue')
    prediction = plt.scatter(np.arange(1, N+1), y_preds, c='red')
    plt.xticks(np.arange(1, N+1))
    plt.xlabel('# Oberservation')
    plt.ylabel('Enrollments')
    title = 'True labels vs. Predicted Labels ({})'.format(model_name)
    plt.title(title)
    plt.legend((original, prediction), ('Original', 'Prediction'))
    plt.show()


# ## Linear regression: A baseline

# In[70]:


import statsmodels.api as sm

X_train_refined = X_train.drop(columns=['row_id'], axis=1)
linear_regression = sm.OLS(y_train, X_train_refined)
linear_regression = linear_regression.fit()


# In[187]:


X_test_refined = X_test.drop(columns=['row_id'], axis=1)
y_preds = linear_regression.predict(X_test_refined)


# In[180]:


calculate_metrics(y_test, y_preds)


# In[188]:


plot_preds(y_test, y_preds, 'Linear Regression')


# In[72]:


print(linear_regression.summary())


# In[97]:


pd.DataFrame(linear_regression.pvalues)    .reset_index()    .rename(columns={'index':'Terms', 0:'p_value'})    .sort_values('p_value')


# ## Model 02: Decision Tree

# In[189]:


from sklearn.tree import DecisionTreeRegressor

dtree = DecisionTreeRegressor(max_depth=5, min_samples_leaf =4, random_state=7)
dtree.fit(X_train_refined, y_train)
y_preds = dtree.predict(X_test_refined)

calculate_metrics(y_test, y_preds)


# In[190]:


plot_preds(y_test, y_preds, 'Decision Tree')


# ## Decision tree visualization

# In[191]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()

export_graphviz(dtree, out_file=dot_data, 
                feature_names=X_train_refined.columns,
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# ## Model 03: `XGBoost`

# In[133]:


import xgboost as xgb


# In[134]:


DM_train = xgb.DMatrix(data=X_train_refined,label=y_train)
DM_test = xgb.DMatrix(data=X_test_refined,label=y_test)


# In[220]:


parameters = {
    'max_depth': 6,
    'objective': 'reg:linear',
    'booster': 'gblinear',
    'n_estimators': 1000,
    'learning_rate': 0.2,
    'gamma': 0.01,
    'random_state': 7,
    'subsample': 1.
}


# In[236]:


parameters = {
    'max_depth': 6,
    'objective': 'reg:linear',
    'booster': 'gblinear',
    'n_estimators': 1000,
    'learning_rate': 0.2,
    'gamma': 0.01,
    'random_state': 7,
    'subsample': 1.
}


# In[237]:


xg_reg = xgb.train(params = parameters, dtrain=DM_train, num_boost_round=8)
y_preds = xg_reg.predict(DM_test)


# In[238]:


calculate_metrics(y_test, y_preds)


# In[239]:


plot_preds(y_test, y_preds, 'XGBoost')


# > I used a `gblinear` booster for XGBoost and XGBoost currently does support feature importances linear models.

# ## Model 04: H2O.ai's AutoML

# In[10]:


import h2o
from h2o.automl import H2OAutoML
h2o.init()


# > To use h2o.ai's utilities on the dataset, the library requires the data to be in **H2OFrame** format. 

# In[14]:


X_train['Enrollments'] = y_train
X_test['Enrollments'] = y_test


# In[16]:


X_train_h2o = h2o.H2OFrame(X_train)
X_test_h2o = h2o.H2OFrame(X_test)


# In[26]:


features = X_train.columns.values.tolist()
target = "Enrollments"


# In[41]:


# Construct the AutoML pipeline
auto_h2o = H2OAutoML()
# Train 
auto_h2o.train(x=features,
               y=target,
               training_frame=X_train_h2o)


# In[42]:


# Overview of how the models performed
auto_h2o.leaderboard


# In[43]:


# Extract the best model from the leaderboard
auto_h2o = auto_h2o.leader


# In[44]:


X_test_temp = X_test.copy()
del X_test_temp['Enrollments']


# In[45]:


# Employing the model to make inference
X_test_h2o_copy = h2o.H2OFrame(X_test_temp)
y_preds = auto_h2o.predict(X_test_h2o_copy)

# Convert the predictions to a native list
y_preds = h2o.as_list(y_preds["predict"])


# In[46]:


calculate_metrics(y_test, y_preds)


# > **XGBoost Still Wins!**
