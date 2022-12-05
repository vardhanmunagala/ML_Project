#!/usr/bin/env python
# coding: utf-8

# # IMPORTS

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

from sklearn import model_selection
from sklearn import metrics, ensemble, linear_model
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore') 


# # DATA LOADING

# In[2]:


train = pd.read_csv(r'C:\Users\vardh\OneDrive\Desktop\walmart_dataset\train.csv')
test = pd.read_csv(r'C:\Users\vardh\OneDrive\Desktop\walmart_dataset\test.csv')
stores = pd.read_csv(r'C:\Users\vardh\OneDrive\Desktop\walmart_dataset\stores.csv')
features = pd.read_csv(r'C:\Users\vardh\OneDrive\Desktop\walmart_dataset\features.csv')
sample_submission = pd.read_csv(r'C:\Users\vardh\OneDrive\Desktop\walmart_dataset\sampleSubmission.csv')


# In[3]:


feature_store = features.merge(stores, how='inner', on = "Store")


# In[4]:


train_df = train.merge(feature_store, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)
test_df = test.merge(feature_store, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by = ['Store','Dept','Date']).reset_index(drop=True)


# In[5]:


train_df.describe()


# In[6]:


test_df.describe()


# In[7]:


feature_store = features.merge(stores, how='inner', on = "Store")

# Converting date column to datetime 
feature_store['Date'] = pd.to_datetime(feature_store['Date'])
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])

# Adding some basic datetime features
feature_store['Day'] = feature_store['Date'].dt.day
feature_store['Week'] = feature_store['Date'].dt.week
feature_store['Month'] = feature_store['Date'].dt.month
feature_store['Year'] = feature_store['Date'].dt.year


# In[8]:


train_df = train.merge(feature_store, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)
test_df = test.merge(feature_store, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by = ['Store','Dept','Date']).reset_index(drop=True)


# In[9]:


df_weeks = train_df.groupby('Week').sum()


# # EDA

# In[10]:


palette = px.colors.qualitative.Safe


# In[11]:


px.line( data_frame = df_weeks, x = df_weeks.index, y = 'Weekly_Sales', 
        labels = {'Weekly_Sales' : 'Weekly Sales', 'x' : 'Weeks' }, 
        title = 'Sales over weeks')


# In[12]:


fig = go.Figure()
fig.add_trace(go.Scatter( x = df_weeks.index, y = df_weeks['MarkDown1'], name = 'MarkDown1', mode = 'lines') )
fig.add_trace(go.Scatter( x = df_weeks.index, y = df_weeks['MarkDown2'], name = 'MarkDown2', mode = 'lines') )
fig.add_trace(go.Scatter( x = df_weeks.index, y = df_weeks['MarkDown3'], name = 'MarkDown3', mode = 'lines') )
fig.add_trace(go.Scatter( x = df_weeks.index, y = df_weeks['MarkDown4'], name = 'MarkDown4', mode = 'lines') )
fig.add_trace(go.Scatter( x = df_weeks.index, y = df_weeks['MarkDown5'], name = 'MarkDown5', mode = 'lines') )
fig.add_trace(go.Scatter( x = df_weeks.index, y = df_weeks['Weekly_Sales'], name = 'Weekly Sales', mode = 'lines') )
fig.update_layout(title = "Sales vs Markdown's", xaxis_title = 'Weeks')


# In[13]:


weekly_sales = train_df.groupby(['Year','Week'], as_index = False).agg({'Weekly_Sales': ['mean', 'median']})
weekly_sales2010 = train_df.loc[train_df['Year']==2010].groupby(['Week']).agg({'Weekly_Sales': ['mean', 'median']})
weekly_sales2011 = train_df.loc[train_df['Year']==2011].groupby(['Week']).agg({'Weekly_Sales': ['mean', 'median']})
weekly_sales2012 = train_df.loc[train_df['Year']==2012].groupby(['Week']).agg({'Weekly_Sales': ['mean', 'median']})


# In[14]:


fig = go.Figure()
fig.add_trace(go.Scatter( x = weekly_sales2010['Weekly_Sales']['mean'].index, y = weekly_sales2010['Weekly_Sales']['mean'], name = 'Mean Sales 2010', mode = 'lines') )
fig.add_trace(go.Scatter( x = weekly_sales2011['Weekly_Sales']['mean'].index, y = weekly_sales2011['Weekly_Sales']['mean'], name = 'Mean Sales 2011', mode = 'lines') )
fig.add_trace(go.Scatter( x = weekly_sales2012['Weekly_Sales']['mean'].index, y = weekly_sales2012['Weekly_Sales']['mean'], name = 'Mean Sales 2012', mode = 'lines') )
fig.add_annotation(text="Thanksgiving", x=47, y=25000, showarrow=False)
fig.add_annotation(text="Christmas", x=51, y=29000, showarrow=False)
fig.update_layout(title = 'Sales 2010, 2011, 2012', xaxis_title = 'Weeks')


# In[15]:


# Converting the temperature to celsius for a better interpretation
train_df['Temperature'] = train_df['Temperature'].apply(lambda x :  (x - 32) / 1.8)
train_df['Temperature'] = train_df['Temperature'].apply(lambda x :  (x - 32) / 1.8)


# In[16]:


train_plt = train_df.sample(frac=0.20)


# In[17]:


px.histogram(train_plt, x='Temperature', y ='Weekly_Sales', color='IsHoliday', marginal='box', opacity= 0.6,
             title = 'Temperature and sales by holiday', color_discrete_sequence=palette)


# In[18]:


px.histogram(train_plt, x='Fuel_Price', y ='Weekly_Sales', color='IsHoliday', marginal='box', opacity= 0.6,
             title='Fuel price and sales by holiday',color_discrete_sequence=palette)


# In[19]:


px.histogram(train_plt, x='CPI', y ='Weekly_Sales', color='IsHoliday', marginal='box', opacity= 0.6,
             title='CPI and sales by holiday',color_discrete_sequence=palette)


# In[20]:


px.histogram(train_plt, x='Unemployment', y ='Weekly_Sales', color='IsHoliday', marginal='box', opacity= 0.6,
             title='Unemployment rate and sales by holiday',color_discrete_sequence=palette)


# In[21]:


sizes= train_plt.groupby('Size').mean()
px.line(sizes, x = sizes.index, y = sizes.Weekly_Sales, 
        title='Store size and sales')


# In[22]:


store_type = pd.concat([stores['Type'], stores['Size']], axis=1)
px.box(store_type, x='Type', y='Size', color='Type', 
       title='Store size and Store type',color_discrete_sequence=palette)


# In[23]:


store_sale = pd.concat([stores['Type'], train_df['Weekly_Sales']], axis=1)
px.box(store_sale.dropna(), x='Type', y='Weekly_Sales', color='Type', 
       title='Store type and sales',color_discrete_sequence=palette)


# In[24]:


depts= train_plt.groupby('Dept').mean().sort_values(by='Weekly_Sales', ascending=False)
bar=px.bar(depts, x = depts.index, y =  depts.Weekly_Sales, title='Departament and sales',color=depts.Weekly_Sales)
bar.update_layout(barmode='group', xaxis={'categoryorder':'total descending'})


# In[25]:


corr = train_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
df_mask = corr.mask(mask).round(2)

fig = ff.create_annotated_heatmap(z=df_mask.to_numpy(), 
                                  x=df_mask.columns.tolist(),
                                  y=df_mask.columns.tolist(),
                                  colorscale=px.colors.diverging.RdBu,
                                  hoverinfo="none", 
                                  showscale=True, ygap=1, xgap=1
                                 )

fig.update_xaxes(side="bottom")

fig.update_layout(
    title_text='Heatmap', 
    title_x=0.5, 
    width=900, 
    height=700,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    xaxis_zeroline=False,
    yaxis_zeroline=False,
    yaxis_autorange='reversed',
    template='plotly_white'
)

for i in range(len(fig.layout.annotations)):
    if fig.layout.annotations[i].text == 'nan':
        fig.layout.annotations[i].text = ""

fig.show()


# In[26]:


weekly_sales_corr = train_df.corr().iloc[2,:]
corr_df = pd.DataFrame(data = weekly_sales_corr, index = weekly_sales_corr.index ).sort_values (by = 'Weekly_Sales', ascending = False)
corr_df = corr_df.iloc[1:]
bar = px.bar(corr_df, x = corr_df.index, y = 'Weekly_Sales', color=corr_df.index, labels={'index':'Featues'},
             title='Feature correlation with sales',color_discrete_sequence=palette)
bar.update_traces(showlegend=False)


# In[27]:


data_train = train_df.copy()
data_test = test_df.copy()


# # FEATURE ENGINEERING

# In[28]:


data_train['Days_to_Thansksgiving'] = (pd.to_datetime(train_df["Year"].astype(str)+"-11-24", format="%Y-%m-%d") - pd.to_datetime(train_df["Date"], format="%Y-%m-%d")).dt.days.astype(int)
data_train['Days_to_Christmas'] = (pd.to_datetime(train_df["Year"].astype(str)+"-12-24", format="%Y-%m-%d") - pd.to_datetime(train_df["Date"], format="%Y-%m-%d")).dt.days.astype(int)


# In[29]:


data_test['Days_to_Thansksgiving'] = (pd.to_datetime(test_df["Year"].astype(str)+"-11-24", format="%Y-%m-%d") - pd.to_datetime(test_df["Date"], format="%Y-%m-%d")).dt.days.astype(int)
data_test['Days_to_Christmas'] = (pd.to_datetime(test_df["Year"].astype(str)+"-12-24", format="%Y-%m-%d") - pd.to_datetime(test_df["Date"], format="%Y-%m-%d")).dt.days.astype(int)


# In[30]:


data_train['SuperBowlWeek'] = train_df['Week'].apply(lambda x: 1 if x == 6 else 0)
data_train['LaborDay'] = train_df['Week'].apply(lambda x: 1 if x == 36 else 0)
data_train['Thanksgiving'] = train_df['Week'].apply(lambda x: 1 if x == 47 else 0)
data_train['Christmas'] = train_df['Week'].apply(lambda x: 1 if x == 52 else 0)


# In[31]:


data_test['SuperBowlWeek'] = test_df['Week'].apply(lambda x: 1 if x == 6 else 0)
data_test['LaborDay'] = test_df['Week'].apply(lambda x: 1 if x == 36 else 0)
data_test['Thanksgiving'] = test_df['Week'].apply(lambda x: 1 if x == 47 else 0)
data_test['Christmas'] = test_df['Week'].apply(lambda x: 1 if x == 52 else 0)


# In[32]:


data_train['MarkdownsSum'] = train_df['MarkDown1'] + train_df['MarkDown2'] + train_df['MarkDown3'] + train_df['MarkDown4'] + train_df['MarkDown5'] 


# In[33]:


data_test['MarkdownsSum'] = test_df['MarkDown1'] + test_df['MarkDown2'] + test_df['MarkDown3'] + test_df['MarkDown4'] + test_df['MarkDown5']


# In[34]:


data_train.isna().sum()[data_train.isna().sum() > 0].sort_values(ascending=False)


# In[35]:


data_test.isna().sum()[data_test.isna().sum() > 0].sort_values(ascending=False)


# In[36]:


data_train.fillna(0, inplace = True)


# In[37]:


data_test['CPI'].fillna(data_test['CPI'].mean(), inplace = True)
data_test['Unemployment'].fillna(data_test['Unemployment'].mean(), inplace = True)


# In[38]:


data_test.fillna(0, inplace = True)


# In[39]:


data_train['IsHoliday'] = data_train['IsHoliday'].apply(lambda x: 1 if x == True else 0)
data_test['IsHoliday'] = data_test['IsHoliday'].apply(lambda x: 1 if x == True else 0)


# In[40]:


data_train['Type'] = data_train['Type'].apply(lambda x: 1 if x == 'A' else (2 if x == 'B' else 3))
data_test['Type'] = data_test['Type'].apply(lambda x: 1 if x == 'A' else (2 if x == 'B' else 3))


# # FEATURE SELECTION

# In[41]:


features = [feature for feature in data_train.columns if feature not in ('Date','Weekly_Sales')]


# In[42]:


X = data_train[features].copy()
y = data_train.Weekly_Sales.copy()


# In[43]:


data_sample = data_train.copy().sample(frac=.25)
X_sample = data_sample[features].copy()
y_sample = data_sample.Weekly_Sales.copy()


# In[44]:


X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_sample, y_sample, random_state=0, test_size=0.15)


# In[45]:


feat_model = xgb.XGBRegressor(random_state=0).fit(X_train, y_train)


# In[46]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(feat_model, random_state=1).fit(X_valid, y_valid)
features = eli5.show_weights(perm, top=len(X_train.columns), feature_names = X_valid.columns.tolist())


# In[47]:


features_weights = eli5.show_weights(perm, top=len(X_train.columns), feature_names = X_valid.columns.tolist())
features_weights


# In[48]:


weights = eli5.show_weights(perm, top=len(X_train.columns), feature_names=X_valid.columns.tolist())
result = pd.read_html(weights.data)[0]
result


# # EVALUATION METRIC

# In[49]:


# Eval metric for the competition
def WMAE(dataset, real, predicted):
    weights = dataset.IsHoliday.apply(lambda x: 5 if x else 1)
    return np.round(np.sum(weights*abs(real-predicted))/(np.sum(weights)), 2)


# In[50]:


X_baseline = X[['Store','Dept','IsHoliday','Size','Week','Type','Year','Day']].copy()


# # DATA SPLITTING

# In[ ]:





# In[51]:


X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_baseline, y, random_state=0, test_size=0.1)


# In[52]:


train_inputs = X_train
train_targets = y_train
val_inputs = X_valid
val_targets = y_valid


# # MODELLING

# In[53]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# In[54]:


model_list = ['Linear Regression', 'Ridge Regression', 'Decision Tree', 'Random Forest', 'Xgboost', 'Extra Trees']
training_wmae_list = []
testing_wmae_list = []


# ### LINEAR REGRESSION

# In[55]:


#build the model
model1 = LinearRegression().fit(train_inputs, train_targets)
#training error
linear_train_wmae = WMAE(train_inputs, model1.predict(train_inputs), train_targets)
#validation error
y_pred = model1.predict(val_inputs)
linear_val_wmae = WMAE(val_inputs, y_pred, val_targets)
#results
print('Training dataset WMAE is', linear_train_wmae)
print('Validation dataset WMAE is', linear_val_wmae)


# ### RIDGE REGRESSION

# In[56]:


model2 = Ridge().fit(train_inputs, train_targets)
#train error
ridge_train_wmae = WMAE(train_inputs, model2.predict(train_inputs), train_targets)

#validation error
y_pred = model2.predict(val_inputs)
ridge_val_wmae = WMAE(val_inputs, y_pred, val_targets)

#results:
print('Training dataset WMAE is', ridge_train_wmae)
print('Validation dataset WMAE is', ridge_val_wmae)


# ### DECISION TREE

# In[57]:


model3 = DecisionTreeRegressor(random_state=0).fit(train_inputs, train_targets)
#train error
tree_train_wmae = WMAE(train_inputs, model2.predict(train_inputs), train_targets)

#validation error
y_pred = model3.predict(val_inputs)
tree_val_wmae = WMAE(val_inputs, y_pred, val_targets)

#results:
print('Training dataset WMAE is', tree_train_wmae)
print('Validation dataset WMAE is', tree_val_wmae)


# In[58]:


def test_params(**params):  
    model = RandomForestRegressor(random_state=0, n_jobs=-1, **params).fit(X_train, y_train)
    train_wmae = WMAE(X_train, y_train, model.predict(X_train))
    val_wmae = WMAE(X_valid, y_valid, model.predict(X_valid))
    return train_wmae, val_wmae


# In[59]:


def test_param_and_plot(param_name, param_values):
    train_errors, val_errors = [], [] 
    for value in param_values:
        params = {param_name: value}
        train_wmae, val_wmae = test_params(**params)
        train_errors.append(train_wmae)
        val_errors.append(val_wmae)
    plt.figure(figsize=(16,8))
    plt.title('Overfitting curve: ' + param_name)
    plt.plot(param_values, train_errors, 'b-o')
    plt.plot(param_values, val_errors, 'r-o')
    plt.xlabel(param_name)
    plt.ylabel('WMAE')
    plt.legend(['Training', 'Validation'])


# In[60]:


test_param_and_plot('max_depth', [10, 15, 20, 25, 30])


# In[61]:


test_param_and_plot('n_estimators', [25, 50, 75, 100])


# In[62]:


test_param_and_plot('min_samples_split', [2, 3, 4, 5])


# In[63]:


test_param_and_plot('min_samples_leaf', [1, 2, 3, 4, 5])


# In[64]:


test_param_and_plot('max_samples', [0.2, 0.4, 0.6, 0.8])


# ### RANDOM FOREST

# In[65]:


#Create the model
rf1 = RandomForestRegressor(random_state=0, n_estimators=100, max_depth=25, min_samples_split=2, min_samples_leaf=1)

# Fit the model
rf1.fit(train_inputs, train_targets)
rf1_train_pred = rf1.predict(train_inputs)

# Compute WMAE on traing data
rf1_train_wmae = WMAE(train_inputs, train_targets, rf1_train_pred)
#print('The WMAE loss for the training set is  {}.'.format(rf1_train_wmae))

rf1_val_preds = rf1.predict(val_inputs)

# Compute WMAE on validation data
rf1_val_wmae = WMAE(val_inputs, val_targets, rf1_val_preds)
#print('The WMAE loss for the validation set is  {}.'.format(rf1_val_wmae))

#results:
print('Training dataset WMAE is', rf1_train_wmae)
print('Validation dataset WMAE is', rf1_val_wmae)


# ### EXTRA TREES

# In[66]:


# Create the model
ETR = ensemble.ExtraTreesRegressor(n_estimators=50, bootstrap = True, random_state = 0)

# Fit the model
ETR.fit(X_train, y_train)

# Compute WMAE on traing data
ETR_train_preds = ETR.predict(X_train)
ETR_train_wmae = WMAE(X_train, y_train, ETR_train_preds)
#print('The WMAE loss for the training set is  {}.'.format(xgb1_train_wmae))


# Compute WMAE on validation data
ETR_val_preds = ETR.predict(X_valid)
ETR_val_wmae = WMAE(X_valid, y_valid, ETR_val_preds)
#print('The WMAE loss for the validation set is  {}.'.format(xgb1_val_wmae))

#results:
print('Training dataset WMAE is', ETR_train_wmae)
print('Validation dataset WMAE is', ETR_val_wmae)


# In[67]:


training_wmae_list.append(linear_train_wmae)
training_wmae_list.append(ridge_train_wmae)
training_wmae_list.append(tree_train_wmae)
training_wmae_list.append(rf1_train_wmae)
training_wmae_list.append(ETR_train_wmae)

testing_wmae_list.append(linear_val_wmae)
testing_wmae_list.append(ridge_val_wmae)
testing_wmae_list.append(tree_val_wmae)
testing_wmae_list.append(rf1_val_wmae)
testing_wmae_list.append(ETR_val_wmae)


# In[68]:


for i in range(5):
    print("For " + model_list[i] + " : Training WMAE is " + str(training_wmae_list[i]) + " & Testing WMAE is " + str(testing_wmae_list[i]))


# In[69]:


test = data_test[['Store','Dept','IsHoliday','Size','Week','Type','Year','Day']].copy()
predict_rf = rf1.predict(test)
predict_etr = ETR.predict(test)


# In[70]:


avg_preds = (predict_rf + predict_etr) / 2


# In[71]:


sample_submission['Weekly_Sales'] = avg_preds
sample_submission.to_csv('submission_rf_etr.csv',index=False)


# In[72]:


X_train.head()

