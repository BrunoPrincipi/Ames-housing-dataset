#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Drop high missing values features
# Impute missing values on categorical features
# Impute missing values on numerical features
# Build a bese model


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





# In[38]:


# The code below creates another copy of the data to do some data exploration
df_version = house.copy()


# ### VERSION 2
# ##### Changes applied on features highly correlated with the label (correlation >=|0.60|):
# ##### Replace outliers for mean value.
# ##### Features distribution with heavy tail or skewed were transformed using np.sqrt() to make the distribution roughly symetrical.

# In[ ]:





# #### Features distribution

# In[39]:


# Check the distribution on the feature with correlation >=|0.60|


# In[40]:


# Check the distribution of the feature
plt.figure(figsize=(12,4))
sns.histplot(df_version['TotalBsmtSF'],kde=True,bins=60)


# In[41]:


# Check the distribution of the feature applying np.sqrt()
plt.figure(figsize=(12,4))
sns.histplot(np.sqrt(df_version['TotalBsmtSF']),kde=True,bins=60)
# should convert to log or sqrt where dist is not normal??


# In[42]:


# Check the distribution of the feature
plt.figure(figsize=(12,4))
sns.histplot(df_version['GrLivArea'],kde=True,bins=60)


# In[43]:


# Check the distribution of the feature applying np.sqrt()
plt.figure(figsize=(12,4))
sns.histplot(np.sqrt(df_version['GrLivArea']),kde=True,bins=60)


# In[44]:


# Check the distribution of the feature
plt.figure(figsize=(12,4))
sns.histplot(df['1stFlrSF'],kde=True,bins=60)


# In[45]:


# Check the distribution of the feature applying np.sqrt()
plt.figure(figsize=(12,4))
sns.histplot(np.sqrt(df['1stFlrSF']),kde=True,bins=60)


# In[46]:


# Check the distribution of the feature
plt.figure(figsize=(12,4))
sns.histplot(df['GarageArea'],kde=True,bins=60)


# In[47]:


# Check the distribution of the feature applying np.sqrt()
plt.figure(figsize=(12,4))
sns.histplot(np.sqrt(df['GarageArea']),kde=True,bins=60)


# In[ ]:





# #### Replace outliers for mean value

# In[48]:


# Method to find outliers base on standard deviation and replace it for the mean value
def std_filter_2(df,feature): 
    std_feature = df[feature].std()    
    filter_output = df[abs(df[feature] - df[feature].mean()) >= 3*std_feature].index.tolist()
    df.loc[filter_output,feature] = df[feature].mean()
    
    return filter_output  # df.loc[filter_output][feature]  


# In[49]:


# On the folowing features the outliers values will be replace for the mean value
        
# TotalBsmtSF            
# GrLivArea

# GarageArea 
# 1stFlrSF 


# In[50]:


df_numerical_final.shape


# In[51]:


# Check the mean of the feature
df_numerical_final['TotalBsmtSF'].mean()


# In[52]:


# Check the mean of the feature
df_numerical_final['GrLivArea'].mean()


# In[53]:


# Check the mean of the feature
df_numerical_final['1stFlrSF'].mean()


# In[54]:


# Check the mean of the feature
df_numerical_final['GarageArea'].mean()


# In[ ]:





# In[55]:


# Applying the method to replace outliers for the mean value
outliers_TotalBsmtSF =std_filter_2(df_numerical_final,'TotalBsmtSF')


# In[56]:


# Checking if the outliers values of the feature were changed for the mean value
df_numerical_final.loc[outliers_TotalBsmtSF,'TotalBsmtSF']


# In[57]:


# Applying the method to replace outliers for the mean value
outliers_GrLivArea =std_filter_2(df_numerical_final,'GrLivArea')


# In[58]:


# Checking if the outliers values of the feature were changed for the mean value
df_numerical_final.loc[outliers_GrLivArea,'GrLivArea']


# In[59]:


# Applying the method to replace outliers for the mean value
outliers_1stFlrSF =std_filter_2(df_numerical_final,'1stFlrSF')


# In[60]:


# Checking if the outliers values of the feature were changed for the mean value
df_numerical_final.loc[outliers_1stFlrSF,'1stFlrSF']


# In[61]:


# Applying the method to replace outliers for the mean value

outliers_GarageArea =std_filter_2(df_numerical_final,'GarageArea')


# In[62]:


# Checking if the outliers values of the feature were changed for the mean value
df_numerical_final.loc[outliers_GarageArea,'GarageArea']


# In[63]:


df_numerical_final.shape


# In[64]:


df_categorical.shape # rows should match


# In[ ]:





# 
# #### Distribution changes

# In[ ]:





# In[65]:


# Transform the features using np.sqrt() to reduce the impact of extreme values
# and make the distribution more symmetrical

df_numerical_final['TotalBsmtSF'] = np.sqrt(df_numerical_final['TotalBsmtSF'])

df_numerical_final['GrLivArea'] = np.sqrt(df_numerical_final['GrLivArea'])

df_numerical_final['1stFlrSF'] = np.sqrt(df_numerical_final['1stFlrSF'])

df_numerical_final['GarageArea'] = np.sqrt(df_numerical_final['GarageArea'])


# In[ ]:





# #### One hot encoding on categorical features - train dataset

# In[66]:


# OneHotEncoder is a sklearn transformer that perform one hot encoding on categorical features
# The code import the packages needed
from sklearn.preprocessing import OneHotEncoder


# In[67]:


# the code below creates an instance of OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore' )


# In[ ]:





# In[68]:


# the code below stores in a variable the output of one hot encoding on categorical features
df_categorical_final = pd.DataFrame(onehot_encoder.fit_transform(df_categorical),
                           columns=onehot_encoder.get_feature_names(),
                          index=df_categorical.index)


# In[69]:


df_categorical_final.shape


# In[ ]:





# In[70]:


df_final = pd.concat([df_numerical_final,df_categorical_final],axis=1)


# In[71]:


df_final.head()


# In[72]:


# X = df_final


# In[73]:


# y = df['SalePrice']


# In[ ]:





# ### <div class="alert alert-block alert-info">
# 

# ### <div class="alert alert-block alert-info">

# ### Test dataset manipulation and filling missing values

# In[74]:


# the code below load the dataset
house_test = pd.read_csv('house_test.csv')


# In[75]:


# the code below creates a copy of the original dataset
dft = house_test.copy()


# In[76]:


# the code below inspects the dataset with .info() method
dft.info()


# In[77]:


# Removing the same features that were removed in the train dataset
dft.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[78]:


# The code below show the categorical features where are missing values
dft.select_dtypes(include='object').isnull().sum().sort_values(ascending=False).head(19)


# In[ ]:





# In[ ]:





# In[ ]:





# ### Missing values on categorical features - test dataset

# In[79]:


# According to the information in the database, the lack of data on some features 
# is due to the non-existence of the feature. Therefore the missing values on categorical features 
#can be filled for example with 'None'
# The code bellow fill the missing values with 'None' on the following features: MasVnrType,
# BsmtQual,BsmtCond,BsmtExposure,BsmtFinType1,
# BsmtFinType2,FireplaceQu,GarageType,GarageFinish,GarageQual,GarageCond,


# In[80]:


# Imputing missing values on categorical features
dft[['MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond']] = dft[['MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond']].fillna('None')


# #### Missing values on 'MSZoning'

# In[81]:


# Filling missing values on 'MSZoning' based on the mode


# In[82]:


dft['MSZoning'].value_counts()


# In[83]:


dft['MSZoning'] = dft['MSZoning'].fillna('RL')


# ####  Missing values on 'Utilities'

# In[84]:


# Filling missing values on 'Utilities' based on the mode


# In[85]:


dft['Utilities'].value_counts()


# In[86]:


dft['Utilities'] = dft['Utilities'].fillna('AllPub')


# #### Missing values on 'Functional'

# In[87]:


# Filling missing values based on the mode


# In[88]:


dft['Functional'].value_counts()


# In[89]:


dft['Functional'] = dft['Functional'].fillna('Typ') 


# #### Missing values on 'KitchenQual'

# In[90]:


# Filling missing values based on the mode considering the 'Neighborhood'


# In[91]:


dft[dft['KitchenQual'].isnull()]['Neighborhood']


# In[92]:


dft[dft['Neighborhood']=='ClearCr']['KitchenQual'].value_counts()


# In[93]:


dft['KitchenQual'] = dft['KitchenQual'].fillna('TA')


# #### Missing values on 'SaleType'

# In[94]:


# Filling missing values based on the mode


# In[95]:


dft['SaleType'].value_counts()


# In[96]:


dft['SaleType'] = dft['SaleType'].fillna('WD')


# #### Missing values on 'Exterior2nd'

# In[97]:


# Filling missing values based on the mode considering the 'Neighborhood'


# In[98]:


dft[dft['Exterior2nd'].isnull()]['Neighborhood']


# In[99]:


dft[dft['Neighborhood']=='Edwards']['Exterior2nd'].value_counts()


# In[100]:


dft['Exterior2nd'] = dft['Exterior2nd'].fillna('Wd Sdng')


# #### Missing values on 'Exterior1st'

# In[101]:


# Filling missing values based on the mode considering the 'Neighborhood'


# In[102]:


dft[dft['Exterior1st'].isnull()]['Neighborhood']


# In[103]:


dft[dft['Neighborhood']=='Edwards']['Exterior1st'].value_counts()


# In[104]:


dft['Exterior1st'] = dft['Exterior1st'].fillna('Wd Sdng')


# In[105]:


# dft.select_dtypes(include='object').isnull().sum().sort_values(ascending=False).head(19)


# ### Missing values on numerical features - test dataset

# In[106]:


# According to the information in the database, the lack of data on some features 
# is due to the non-existence of the feature. Therefore on thr numerical features the value 
# can be filled for example with (0.0)


# In[107]:


# Imputing missing values on numerical features
dft[['MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','GarageCars','GarageArea','BsmtFullBath','BsmtHalfBath']] = dft[['MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','GarageCars','GarageArea','BsmtFullBath','BsmtHalfBath']].fillna(0.0)


# #### Missing values on 'GarageYrBlt'

# In[ ]:





# In[108]:


# Filling missing values on the feature 'GarageYrBlt' based on the feature 'GarageCond'
dft[dft['GarageYrBlt'].isnull()]['GarageCond'].value_counts()


# In[109]:


dft['GarageYrBlt'] = dft['GarageYrBlt'].fillna(0.0)


# #### Missing numerical values on 'LotFrontage' predicted with IterativeImputer - test data

# In[110]:


# the code below stores in a variable the categorical features of the dataframe
dft_categorical = dft.select_dtypes(include='object')


# In[111]:


# the code below stores in a variable the numerical features of the dataframe
dft_numerical = dft.select_dtypes(exclude='object')


# In[112]:


# the code below stores in a variable the predicted missing values on numerical features
Xt_numerical = iterative_imputer.transform(dft_numerical)


# In[113]:


Xt_numerical


# In[114]:


# the code below creates a dataframe with the numerical features without missing values
dft_numerical_final = pd.DataFrame(Xt_numerical,columns=dft_numerical.columns,index=dft_numerical.index)


# In[115]:


dft_numerical_final.info()


# In[ ]:





# In[116]:


# The code below show the numerical features where are missing values
dft.select_dtypes(exclude='object').isnull().sum().sort_values(ascending=False).head(12)


# #### VERSION 2 (test set)

# #### Replicate the changes made on the train set for version 2
# 

# #### Replace outliers for the mean value

# In[117]:


dft_numerical_final.shape


# In[118]:


# Check the mean of the feature
dft_numerical_final['TotalBsmtSF'].mean()


# In[119]:


# Check the mean of the feature
dft_numerical_final['GrLivArea'].mean()


# In[120]:


# Check the mean of the feature
dft_numerical_final['1stFlrSF'].mean()


# In[121]:


# Check the mean of the feature
dft_numerical_final['GarageArea'].mean()


# In[122]:


# Applying the method to replace outliers for the mean value

outliers_TotalBsmtSF_2 =std_filter_2(dft_numerical_final,'TotalBsmtSF')


# In[123]:


# Check that outliers have the new value mean of the feature
dft_numerical_final.loc[outliers_TotalBsmtSF_2,'TotalBsmtSF'] ### df.loc[index,feature]


# In[124]:


outliers_GrLivArea_2 =std_filter_2(dft_numerical_final,'GrLivArea')


# In[125]:


# Check that outliers have the new value mean of the feature
dft_numerical_final.loc[outliers_GrLivArea_2,'GrLivArea']


# In[126]:


outliers_1stFlrSF_2 =std_filter_2(dft_numerical_final,'1stFlrSF')


# In[127]:


# Check that outliers have the new value mean of the feature
dft_numerical_final.loc[outliers_1stFlrSF_2,'1stFlrSF']


# In[128]:


# Check that outliers have the new value mean of the feature
outliers_GarageArea_2 =std_filter_2(dft_numerical_final,'GarageArea')


# In[129]:


dft_numerical_final.loc[outliers_GarageArea_2,'GarageArea']


# In[130]:


dft_numerical_final.shape


# In[131]:


dft_categorical.shape


# #### Change the distribution 

# In[132]:


# Transform the features using np.sqrt() to reduce the impact of extreme values
# and make the distribution more symmetrical

dft_numerical_final['TotalBsmtSF'] = np.sqrt(dft_numerical_final['TotalBsmtSF'])

dft_numerical_final['GrLivArea'] = np.sqrt(dft_numerical_final['GrLivArea'])

dft_numerical_final['1stFlrSF'] = np.sqrt(dft_numerical_final['1stFlrSF'])

dft_numerical_final['GarageArea'] = np.sqrt(dft_numerical_final['GarageArea'])


# In[ ]:





# #### OneHotEncoding on categorical features - test dataset

# In[133]:


# the code below stores in a variable the output of OneHotEncoding on categorical features
dft_categorical_final = pd.DataFrame(onehot_encoder.transform(dft_categorical),
                        columns=onehot_encoder.get_feature_names(),
                        index=dft_categorical.index)


# In[134]:


dft_categorical_final.shape


# In[135]:


# The code below concatenate the numerical features dataframe without missing values 
# with the categorical features dataframe transformed using OneHotEncoding
dft_final = pd.concat([dft_numerical_final,dft_categorical_final],axis=1)


# In[136]:


dft_final.head()


# In[137]:


dft_final.shape


# In[138]:


df_final.head()


# In[139]:


df_final.shape # including ['SalePrice']


# In[140]:


df_final['SalePrice']


# In[ ]:





# In[ ]:





# ### <div class="alert alert-block alert-info">

# ### <div class="alert alert-block alert-info">

# In[ ]:





# ### Models

# In[141]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# In[ ]:





# In[142]:


# The code below store the features on the variable X 
X = df_final.drop('SalePrice',axis=1)


# In[143]:


# The code below store the label on the variable y 
y = df_final['SalePrice']


# In[144]:


# The code below perform the split of the data into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[145]:


sum(X_train.isnull().any())


# In[146]:


# Check the shape of the splits to fit the models
print('X_train', X_train.shape)
print('X_test', X_test.shape)
print('y_train', y_train.shape)
print('y_test', y_test.shape)


# In[ ]:





# #### StandardScaler

# In[147]:


# Create an instance of StandardScaler
std = StandardScaler()


# In[148]:


# The code below scale the features
X_train_scaled = std.fit_transform(X_train)


# In[149]:


X_test_scaled = std.transform(X_test)


# #### LinearRegression

# In[150]:


# Only without scaling


# In[151]:


# The code below creates an instance of the model
linear_regression_1 = LinearRegression()


# In[152]:


# The code below fit the model without scaling X_train
linear_regression_1.fit(X_train,y_train)


# In[153]:


# The code below creates predictions without scaling X_test
pred_linear_regression_1 = linear_regression_1.predict(X_test)


# In[154]:


# The code below fit the model with the train scaled data to find parameters between features and label
# linear_regression_1.fit(X_train_scaled,y_train)


# In[155]:


# The code below creates predictions using test data based on the parameters learned 
# on the fit method
# pred_linear_regression_1 = linear_regression_1.predict(X_test_scaled)


# In[156]:


# Root Mean Squared Error (RMSE) between the truth and the predictions
RMSE_linear_V2 = np.sqrt(mean_squared_error(y_test,pred_linear_regression_1))


# In[157]:


RMSE_linear_V2 # RMSE_linear_V2


# In[158]:


np.sqrt(mean_squared_error(y_test,pred_linear_regression_1))/df['SalePrice'].mean()


# In[159]:


df['SalePrice'].describe()


# In[ ]:





# In[ ]:





# #### ElasticNet

# In[160]:


# Only scaled data, without scaling take to much time


# In[161]:


# The code below creates a grid to combine parameters
param_grid = {'alpha':[.3,.5,1,2],'l1_ratio':[.5,.65,.77,.87,.99]}


# In[162]:


elastic_net = ElasticNet(max_iter=1000000,random_state=101)


# In[163]:


grid_elastic_net = GridSearchCV(elastic_net, param_grid=param_grid, scoring='neg_mean_squared_error',cv=15,verbose=1)


# In[164]:


# Fit the model scaling X_train
grid_elastic_net.fit(X_train_scaled,y_train)


# In[165]:


# grid_elastic_net.fit(X_train,y_train)


# In[166]:


# Fit the model without scaling X_train
# grid_elastic_net.fit(X_train,y_train)


# In[167]:


grid_elastic_net.best_params_


# In[168]:


pred_grid = grid_elastic_net.predict(X_test_scaled)


# In[169]:


# The code below creates predictions based on X_test without scaling
# pred_grid = grid_elastic_net.predict(X_test)


# In[200]:


# Root Mean Squared Error (RMSE) between the truth and the predictions
RMSE_elastic_V2 = np.sqrt(mean_squared_error(y_test,pred_grid))


# In[171]:


np.sqrt(mean_squared_error(y_test,pred_grid))/df['SalePrice'].mean()


# In[201]:


RMSE_elastic_V2


# In[ ]:





# In[ ]:





# #### RandomForestRegressor

# In[173]:


# very similar result, use scaled data


# In[174]:


rf_model = RandomForestRegressor(random_state=101)


# In[217]:


param_grid1 = {'n_estimators':[100,150,200], # number of trees inside the classifier         
               'max_samples':[0.85,0.9], # number of samples to draw from X
               'max_depth': [10,20,25], # the depth of the tree
              'max_features':[30,100,120]} # number of features


# In[176]:


grid_rf = GridSearchCV(rf_model,param_grid1)


# In[177]:


grid_rf.fit(X_train_scaled, y_train)


# In[178]:


# grid_rf.fit(X_train, y_train)


# In[179]:


grid_rf.best_params_


# In[180]:


grid_rf_prediction = grid_rf.predict(X_test_scaled)


# In[181]:


# grid_rf_prediction = grid_rf.predict(X_test)


# In[182]:


# Root Mean Squared Error (RMSE) between the truth and the predictions
RMSE_randomf_V2 = np.sqrt(mean_squared_error(y_test,grid_rf_prediction))


# In[183]:


RMSE_randomf_V2


# In[ ]:





# In[184]:


np.sqrt(mean_squared_error(y_test,grid_rf_prediction))/df['SalePrice'].mean()


# In[ ]:





# #### GradientBoostingRegressor

# In[185]:


# very similar result, use scaled data


# In[186]:


from sklearn.ensemble import GradientBoostingRegressor


# In[187]:


param_grid2 = {'n_estimators':[50,100,200],
             'learning_rate':[.03,.05,.1],
             'max_depth':[5,7,9],
               'subsample':[.6],
              'max_features':[15,30,90,130,220]}


# In[188]:


gbr_model = GradientBoostingRegressor(random_state=101)


# In[189]:


gbr_grid = GridSearchCV(gbr_model,param_grid2)


# In[190]:


gbr_grid.fit(X_train_scaled,y_train)


# In[191]:


gbr_grid.best_params_


# In[192]:


gbr_prediction = gbr_grid.predict(X_test_scaled)


# In[193]:


# Root Mean Squared Error (RMSE) between the truth and the predictions
RMSE_gradient_V2 = np.sqrt(mean_squared_error(y_test,gbr_prediction))


# In[194]:


RMSE_gradient_V2


# In[ ]:





# In[195]:


# Ratio (RMSE)/df['SalePrice'].mean()
np.sqrt(mean_squared_error(y_test,gbr_prediction))/df['SalePrice'].mean()


# In[ ]:





# In[202]:


result_table = {'Linear Regression':[RMSE_linear_V2],'Elastic Net':[RMSE_elastic_V2],'Random Forest':[RMSE_randomf_V2],'Gradient Boosting':[RMSE_gradient_V2]}


# In[ ]:





# In[205]:


result_table1 = pd.DataFrame(data=result_table,index = ['Version 2'])


# In[206]:


result_table1


# ### <div class="alert alert-block alert-info">

# ### <div class="alert alert-block alert-info">

# #### Submission to Kaggle

# In[ ]:





# In[207]:


# Check the shape of the test set
dft_final.shape


# In[208]:


df_final.shape


# In[209]:


# Check the shape of the train set
X.shape


# In[ ]:


# std = StandardScaler()


# In[210]:


# The code below scale the features on the test set with the same scaler used on the train set
dft_final_scaled = std.transform(dft_final)


# In[ ]:


# The code below calculates predictions based on the parameters learned fitting the train 
# data to the estimator
# kaggle_prediction = linear_regression_1.predict(dft_final_scaled) # LinearRegression


# In[ ]:


# kaggle_prediction = grid_elastic_net.predict(dft_final_scaled) # ElasticNet


# In[ ]:


# kaggle_prediction = grid_rf.predict(dft_final_scaled) # RandomForestRegressor


# In[212]:


# kaggle_prediction = gbr_grid.predict(dft_final_scaled) # GradientBoosting


# In[213]:


# submission = pd.DataFrame({ 'Id': house_test['Id'].values, 'SalePrice': kaggle_prediction })


# In[214]:


# submission.head()


# In[215]:


# submission['SalePrice'] = submission['SalePrice'].astype(int)


# In[216]:


# submission.to_csv('subm/HOUSE_VERSION_2_gbr.csv',index=False)


# In[ ]:




