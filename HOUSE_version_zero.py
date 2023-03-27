#!/usr/bin/env python
# coding: utf-8

# ## The Ames Housing dataset (Kaggle)

# In[2]:


# The code import the packages needed
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno


# In[3]:


# the code below load the dataset
house = pd.read_csv('house_train.csv')


# In[4]:


# the code below creates a copy of the original dataset
df = house.copy()


# In[5]:


# the code below displays the first 5 rows of the database
# Label would be 'SalePrice'
df.head()


# In[6]:


# the code below inspects the dataset with .info() method
# it shows the feature names, the count of non-null values on each feature, 
# and the type of each feature
df.info()


# In[7]:


# the code below inspects how many rows and columns are in the dataset
df.shape


# In[8]:


# the code below displays how many missing values each column has
df.isnull().sum().sort_values(ascending=False).head(25)


# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


# the code bellow shows the proportion of missing values by features
(df.isnull().sum()/len(df)).sort_values(ascending=False).head(20)


# In[10]:


# The following graph shows the missing values by features
msno.matrix(df)


# In[11]:


# The code bellow shows if there are any duplicated row
duplicates = df.duplicated()


# In[12]:


sum(duplicates)


# In[13]:


# The code below shows the linear correlation between 'SalePrice' and the numerical features
df.corr()['SalePrice'].sort_values(ascending=False)


# ## Train dataset manipulation and filling missing values

# #### Dropping features with high proportion of missing values

# In[14]:


# Features that have a proportion greater than 0.4 of missing values are dropped
df.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[ ]:





# #### Imputing missing values on categorical features

# In[15]:


#### According to the database description, the 'nan' values in some features 
#### are due to the non-existence of the feature.
#### Filling missing values with 'None' on the following features:
#### 'BsmtFinType1','BsmtFinType2','BsmtQual','BsmtCond','BsmtExposure','FireplaceQu',
#### 'MasVnrType','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'


# In[16]:


df[['BsmtFinType1','BsmtFinType2','BsmtQual','BsmtCond','BsmtExposure','FireplaceQu','MasVnrType','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']] = df[['BsmtFinType1','BsmtFinType2','BsmtQual','BsmtCond','BsmtExposure','FireplaceQu','MasVnrType','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']].fillna('None')


# In[ ]:





# #### Imputing the mode  on 'Electrical'  missing value

# In[17]:


df['Electrical'].value_counts()


# In[18]:


df['Electrical'] = df['Electrical'].fillna('SBrkr')


# In[19]:


# df.info()


# #### Imputing missing values on numerical features

# #### 'GarageYrBlt'

# In[20]:


# Imputing the mode of 'GarageYrBlt' for the top 3 'Neighborhood' where ['GarageYrBlt'].isnull()


# In[21]:


df[df['GarageYrBlt'].isnull()]['Neighborhood'].value_counts()


# In[22]:


# mode of 'GarageYrBlt' for the top 3 'Neighborhood' where ['GarageYrBlt'].isnull()
df[(df['Neighborhood']== 'Edwards') | (df['Neighborhood']== 'OldTown') | (df['Neighborhood']=='BrkSide')]['GarageYrBlt'].value_counts()


# In[23]:


df['GarageYrBlt'] = df['GarageYrBlt'].fillna(1950.0)


# #### 'MasVnrArea'

# In[24]:


df[df['MasVnrArea'].isnull()]['MasVnrType']


# In[25]:


# Impute 0.0 on 'MasVnrArea' for the missing values base on the 'MasVnrType'
df['MasVnrArea'] = df['MasVnrArea'].fillna(0.0)


# In[26]:


# df.info()


# In[ ]:





# #### Impute missing numerical values on 'LotFrontage' - train dataset 

# In[27]:


### IterativeImputer is a sklearn predictor that perform a regression model to predict 
### missing values on numerical features


# In[28]:


# The code import the packages needed
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# In[29]:


# the code below creates an instance of IterativeImputer
iterative_imputer = IterativeImputer(max_iter=10000)


# In[30]:


# the code below stores in a variable the numerical features of the dataframe
df_numerical = df.select_dtypes(exclude='object')


# In[31]:


# the code below drop the label to not use it as an imput to predict 'LotFrontage'
df_numerical_no_label = df_numerical.drop('SalePrice',axis=1)


# In[32]:


# the code below stores in a variable the categorical features of the dataframe
df_categorical = df.select_dtypes(include='object')


# In[33]:


# the code below stores in a variable the predicted missing values on numerical features
X_numerical = iterative_imputer.fit_transform(df_numerical_no_label)


# In[34]:


# the code below creates a dataframe with the numerical features without missing values,
# and without the label 'SalePrice'
df_numerical_final = pd.DataFrame(X_numerical,columns=df_numerical_no_label.columns,index=df_numerical_no_label.index)


# In[35]:


df_numerical_final['SalePrice'] = df_numerical['SalePrice']


# In[36]:


df_numerical_final.head()


# In[37]:


# df_numerical_final.info()


# In[ ]:





# In[ ]:





# #### One hot encoding on categorical features - train dataset

# In[38]:


# OneHotEncoder is a sklearn transformer that perform one hot encoding on categorical features
# The code import the packages needed
from sklearn.preprocessing import OneHotEncoder


# In[39]:


# the code below creates an instance of OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore' )


# In[ ]:





# In[40]:


# the code below stores in a variable the output of one hot encoding on categorical features
df_categorical_final = pd.DataFrame(onehot_encoder.fit_transform(df_categorical),
                           columns=onehot_encoder.get_feature_names(),
                          index=df_categorical.index)


# In[41]:


df_categorical_final.shape


# In[ ]:





# In[42]:


df_final = pd.concat([df_numerical_final,df_categorical_final],axis=1)


# In[43]:


df_final.head()


# In[44]:


# X = df_final


# In[45]:


# y = df['SalePrice']


# In[ ]:





# ### <div class="alert alert-block alert-info">
# 

# ### <div class="alert alert-block alert-info">

# ### Test dataset manipulation and filling missing values

# In[46]:


# the code below load the dataset
house_test = pd.read_csv('house_test.csv')


# In[47]:


# the code below creates a copy of the original dataset
dft = house_test.copy()


# In[48]:


# the code below inspects the dataset with .info() method
dft.info()


# In[49]:


# Removing the same features that were removed in the train dataset
dft.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[50]:


# The code below show the categorical features where are missing values
dft.select_dtypes(include='object').isnull().sum().sort_values(ascending=False).head(19)


# In[ ]:





# In[ ]:





# In[ ]:





# ### Missing values on categorical features - test dataset

# In[51]:


# According to the information in the database, the lack of data on some features 
# is due to the non-existence of the feature. Therefore the missing values on categorical features 
# can be filled for example with 'None'
# The code bellow fill the missing values with 'None' on the following features: MasVnrType,
# BsmtQual,BsmtCond,BsmtExposure,BsmtFinType1,
# BsmtFinType2,FireplaceQu,GarageType,GarageFinish,GarageQual,GarageCond,


# In[52]:


# Imputing missing values on categorical features
dft[['MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond']] = dft[['MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond']].fillna('None')


# #### Missing values on 'MSZoning'

# In[53]:


# Filling missing values on 'MSZoning' based on the mode


# In[54]:


dft['MSZoning'].value_counts()


# In[55]:


dft['MSZoning'] = dft['MSZoning'].fillna('RL')


# ####  Missing values on 'Utilities'

# In[56]:


# Filling missing values on 'Utilities' based on the mode


# In[57]:


dft['Utilities'].value_counts()


# In[58]:


dft['Utilities'] = dft['Utilities'].fillna('AllPub')


# #### Missing values on 'Functional'

# In[59]:


# Filling missing values based on the mode


# In[60]:


dft['Functional'].value_counts()


# In[61]:


dft['Functional'] = dft['Functional'].fillna('Typ') 


# #### Missing values on 'KitchenQual'

# In[62]:


# Filling missing values based on the mode considering the 'Neighborhood'


# In[63]:


dft[dft['KitchenQual'].isnull()]['Neighborhood']


# In[64]:


dft[dft['Neighborhood']=='ClearCr']['KitchenQual'].value_counts()


# In[65]:


dft['KitchenQual'] = dft['KitchenQual'].fillna('TA')


# #### Missing values on 'SaleType'

# In[66]:


# Filling missing values based on the mode


# In[67]:


dft['SaleType'].value_counts()


# In[68]:


dft['SaleType'] = dft['SaleType'].fillna('WD')


# #### Missing values on 'Exterior2nd'

# In[69]:


# Filling missing values based on the mode considering the 'Neighborhood'


# In[70]:


dft[dft['Exterior2nd'].isnull()]['Neighborhood']


# In[71]:


dft[dft['Neighborhood']=='Edwards']['Exterior2nd'].value_counts()


# In[72]:


dft['Exterior2nd'] = dft['Exterior2nd'].fillna('Wd Sdng')


# #### Missing values on 'Exterior1st'

# In[73]:


# Filling missing values based on the mode considering the 'Neighborhood'


# In[74]:


dft[dft['Exterior1st'].isnull()]['Neighborhood']


# In[75]:


dft[dft['Neighborhood']=='Edwards']['Exterior1st'].value_counts()


# In[76]:


dft['Exterior1st'] = dft['Exterior1st'].fillna('Wd Sdng')


# In[77]:


# dft.select_dtypes(include='object').isnull().sum().sort_values(ascending=False).head(19)


# ### Missing values on numerical features - test dataset

# In[78]:


# According to the information in the database, the lack of data on some features 
# is due to the non-existence of the feature. Therefore on thr numerical features the value 
# can be filled for example with (0.0)


# In[79]:


# Imputing missing values on numerical features
dft[['MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','GarageCars','GarageArea','BsmtFullBath','BsmtHalfBath']] = dft[['MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','GarageCars','GarageArea','BsmtFullBath','BsmtHalfBath']].fillna(0.0)


# #### Missing values on 'GarageYrBlt'

# In[ ]:





# In[80]:


# Filling missing values on the feature 'GarageYrBlt' based on the feature 'GarageCond'
dft[dft['GarageYrBlt'].isnull()]['GarageCond'].value_counts()


# In[81]:


dft['GarageYrBlt'] = dft['GarageYrBlt'].fillna(0.0)


# #### Missing numerical values on 'LotFrontage' predicted with IterativeImputer - test data

# In[82]:


# the code below stores in a variable the categorical features of the dataframe
dft_categorical = dft.select_dtypes(include='object')


# In[83]:


# the code below stores in a variable the numerical features of the dataframe
dft_numerical = dft.select_dtypes(exclude='object')


# In[84]:


# the code below stores in a variable the predicted missing values on numerical features
Xt_numerical = iterative_imputer.transform(dft_numerical)


# In[85]:


Xt_numerical


# In[86]:


# the code below creates a dataframe with the numerical features without missing values
dft_numerical_final = pd.DataFrame(Xt_numerical,columns=dft_numerical.columns,index=dft_numerical.index)


# In[87]:


dft_numerical_final.info()


# In[ ]:





# In[88]:


# The code below show the numerical features where are missing values
dft.select_dtypes(exclude='object').isnull().sum().sort_values(ascending=False).head(12)


# #### OneHotEncoding on categorical features - test dataset

# In[89]:


# the code below stores in a variable the output of OneHotEncoding on categorical features
dft_categorical_final = pd.DataFrame(onehot_encoder.transform(dft_categorical),
                        columns=onehot_encoder.get_feature_names(),
                        index=dft_categorical.index)


# In[90]:


dft_categorical_final.shape


# In[91]:


# The code below concatenate the numerical features dataframe without missing values 
# with the categorical features dataframe transformed using OneHotEncoding
dft_final = pd.concat([dft_numerical_final,dft_categorical_final],axis=1)


# In[92]:


dft_final.head()


# In[93]:


dft_final.shape


# In[179]:


# df_final.head()


# In[95]:


df_final.shape # including ['SalePrice']


# In[ ]:





# ### <div class="alert alert-block alert-info">

# ### <div class="alert alert-block alert-info">

# In[ ]:





# ### Models

# In[97]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# In[ ]:





# In[98]:


# The code below store the features on the variable X 
X = df_final.drop('SalePrice',axis=1)


# In[99]:


# The code below store the label on the variable y 
y = df_final['SalePrice']


# In[100]:


# The code below perform the split of the data into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[101]:


sum(X_train.isnull().any())


# In[102]:


# Check the shape of the splits to fit the models
print('X_train', X_train.shape)
print('X_test', X_test.shape)
print('y_train', y_train.shape)
print('y_test', y_test.shape)


# In[ ]:





# #### StandardScaler

# In[103]:


# Create an instance of StandardScaler
std = StandardScaler()


# In[104]:


# The code below scale the features
X_train_scaled = std.fit_transform(X_train)


# In[105]:


X_test_scaled = std.transform(X_test)


# #### LinearRegression

# In[107]:


# The code below creates an instance of the model
linear_regression_1 = LinearRegression()


# In[108]:


# The code below fit the model without scaling X_train
linear_regression_1.fit(X_train,y_train)


# In[109]:


# The code below creates predictions without scaling X_test
pred_linear_regression_1 = linear_regression_1.predict(X_test)


# In[110]:


# The code below fit the model with the train scaled data to find parameters between features and label
# linear_regression_1.fit(X_train_scaled,y_train)


# In[111]:


# The code below creates predictions using test data based on the parameters learned 
# on the fit method
# pred_linear_regression_1 = linear_regression_1.predict(X_test_scaled)


# In[112]:


# Root Mean Squared Error (RMSE) between the truth and the predictions
RMSE_linear_V0 = np.sqrt(mean_squared_error(y_test,pred_linear_regression_1))


# In[113]:


RMSE_linear_V0 


# In[114]:


np.sqrt(mean_squared_error(y_test,pred_linear_regression_1))/df['SalePrice'].mean()


# In[115]:


df['SalePrice'].describe()


# In[ ]:





# In[ ]:





# #### ElasticNet

# In[120]:


# Only scaled data, without scaling take to much time


# In[121]:


# The code below creates a grid to combine parameters
param_grid = {'alpha':[.3,.5,1,2],'l1_ratio':[.5,.65,.77,.87,.99]}


# In[122]:


elastic_net = ElasticNet(max_iter=1000000,random_state=101)


# In[123]:


grid_elastic_net = GridSearchCV(elastic_net, param_grid=param_grid, scoring='neg_mean_squared_error',cv=15,verbose=1)


# In[124]:


# Fit the model scaling X_train
grid_elastic_net.fit(X_train_scaled,y_train)


# In[125]:


# grid_elastic_net.fit(X_train,y_train)


# In[126]:


# Fit the model without scaling X_train
# grid_elastic_net.fit(X_train,y_train)


# In[127]:


grid_elastic_net.best_params_


# In[128]:


pred_grid = grid_elastic_net.predict(X_test_scaled)


# In[129]:


# The code below creates predictions based on X_test without scaling
# pred_grid = grid_elastic_net.predict(X_test)


# In[130]:


# Root Mean Squared Error (RMSE) between the truth and the predictions
RMSE_elastic_V0 = np.sqrt(mean_squared_error(y_test,pred_grid))


# In[131]:


RMSE_elastic_V0 


# In[132]:


np.sqrt(mean_squared_error(y_test,pred_grid))/df['SalePrice'].mean()


# In[ ]:





# #### RandomForestRegressor

# In[135]:


# very similar result, use scaled data run faster


# In[136]:


rf_model = RandomForestRegressor(random_state=101)


# In[137]:


param_grid1 = {'n_estimators':[100,150,200], # number of trees inside the classifier         
               'max_samples':[0.85,0.9],
               'max_depth': [10,20,25], # the depth of the tree
              'max_features':[30,100,120]} 


# In[138]:


grid_rf = GridSearchCV(rf_model,param_grid1)


# In[139]:


grid_rf.fit(X_train_scaled, y_train)


# In[140]:


# grid_rf.fit(X_train, y_train)


# In[141]:


grid_rf.best_params_


# In[142]:


grid_rf_prediction = grid_rf.predict(X_test_scaled)


# In[143]:


# grid_rf_prediction = grid_rf.predict(X_test)


# In[144]:


# Root Mean Squared Error (RMSE) between the truth and the predictions
RMSE_randomf_V0 = np.sqrt(mean_squared_error(y_test,grid_rf_prediction))


# In[145]:


RMSE_randomf_V0 


# In[146]:


np.sqrt(mean_squared_error(y_test,grid_rf_prediction))/df['SalePrice'].mean()


# #### GradientBoostingRegressor

# In[148]:


from sklearn.ensemble import GradientBoostingRegressor


# In[149]:


param_grid2 = {'n_estimators':[50,100,200],
             'learning_rate':[.03,.05,.1],
             'max_depth':[5,7,9],
               'subsample':[.6],
              'max_features':[15,30,90,130,220]}


# In[150]:


gbr_model = GradientBoostingRegressor(random_state=101)


# In[151]:


gbr_grid = GridSearchCV(gbr_model,param_grid2)


# In[152]:


gbr_grid.fit(X_train_scaled,y_train)


# In[153]:


gbr_grid.best_params_


# In[154]:


gbr_prediction = gbr_grid.predict(X_test_scaled)


# In[168]:


# Root Mean Squared Error (RMSE) between the truth and the predictions
RMSE_gradient_V0 = np.sqrt(mean_squared_error(y_test,gbr_prediction))


# In[156]:


# Ratio between Root Mean Squared Error (RMSE)/df['SalePrice'].mean()
np.sqrt(mean_squared_error(y_test,gbr_prediction))/df['SalePrice'].mean()


# In[169]:


RMSE_gradient_V0


# In[ ]:





# In[170]:


result_table = {'Linear Regression':[RMSE_linear_V0],'Elastic Net':[RMSE_elastic_V0],'Random Forest':[RMSE_randomf_V0],'Gradient Boosting':[RMSE_gradient_V0]}


# In[173]:


result_table1 = pd.DataFrame(data=result_table,index = ['Version 0'])


# In[174]:


result_table1


# In[ ]:





# ### <div class="alert alert-block alert-info">

# ### <div class="alert alert-block alert-info">

# In[161]:





# #### Submission to Kaggle

# In[ ]:





# In[176]:


# Check the shape of the test set
dft_final.shape


# In[177]:


df_final.shape


# In[178]:


# Check the shape of the train set
X.shape


# In[ ]:


# std = StandardScaler()


# In[180]:


# The code below scale the features on the test set with the same scaler used on the train set
dft_final_scaled = std.transform(dft_final)


# In[ ]:


# The code below calculates predictions based on the parameters learned fitting the train 
# data to the estimator 
# kaggle_prediction = linear_regression_1.predict(dft_final_scaled) # LinearRegression


# In[ ]:


# The code below calculates predictions based on the parameters learned fitting the train 
# data to the estimator 
# kaggle_prediction = grid_elastic_net.predict(dft_final_scaled) # ElasticNet


# In[181]:


# The code below calculates predictions based on the parameters learned fitting the train 
# data to the estimator 
# kaggle_prediction = grid_rf.predict(dft_final_scaled) # RandomForestRegressor


# In[182]:


# The code below calculates predictions based on the parameters learned fitting the train 
# data to the estimator 
# kaggle_prediction = gbr_grid.predict(dft_final_scaled) # GradientBoosting


# In[183]:


# submission = pd.DataFrame({ 'Id': house_test['Id'].values, 'SalePrice': kaggle_prediction })


# In[184]:


# submission.head()


# In[185]:


# submission['SalePrice'] = submission['SalePrice'].astype(int)


# In[186]:


# submission.to_csv('subm/HOUSE_VERSION_0_gbr.csv',index=False)


# In[ ]:




