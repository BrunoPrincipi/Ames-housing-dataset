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





# ### VERSION 1 
# #### Drop features high correlated between each other / dorp features low correlated with the label 

# In[38]:


df.corr()['SalePrice'].sort_values()


# In[39]:


# The code below creates another copy of the data to do some data exploration
df_version = house.copy()


# In[40]:


# df_version.info()


# In[41]:


# plt.figure(figsize=(18,10))
# sns.heatmap(df.corr(),annot=True)


# In[42]:


# the code below stores in a variable a correlation matrix with the absolute values 
# of the df DataFrame
corr_matrix = df.corr()#.abs()


# In[43]:


# The code below stores in a variable the values on the upper triangle above the diagonal
# formed by ones of the corr_matrix 
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.abs().shape), k=1).astype(np.bool_))


# In[44]:


# The code below stores in a variable the correlation values on upper that are greater
# than the threshold set
high_collinearity = [column for column in upper.columns if any(upper[column] > 0.80)]


# In[45]:


high_collinearity


# In[ ]:





# In[46]:


# The code below set a threshold to define the low correlated features respect to the label
low_correlated_to_drop = corr_matrix[(corr_matrix['SalePrice'] > -0.05) & (corr_matrix['SalePrice'] < 0.30)].index


# In[47]:


low_correlated_to_drop


# #### Applying changes version 1

# In[48]:


df_numerical_final.shape


# In[49]:


# The following code removes numerical features that are highly correlated with each other

df_numerical_final = df_numerical_final.drop(high_collinearity,axis=1)


# In[50]:


# The following code removes numerical features that are low correlated with the label

df_numerical_final = df_numerical_final.drop(low_correlated_to_drop,axis=1)


# In[51]:


df_numerical_final.shape


# ### VERSION 2
# #### Changes applied on features highly correlated with the label (correlation >=|0.60|):
# #### Delete outliers on train set.
# #### Features distribution with heavy tail or skewed were transformed using np.sqrt() to make the distribution roughly symetrical.

# In[ ]:





# #### Features distribution

# In[52]:


# Check the distribution on the feature with correlation >=|0.60|


# In[53]:


# Check the distribution of the feature
plt.figure(figsize=(12,4))
sns.histplot(df_version['TotalBsmtSF'],kde=True,bins=60)


# In[54]:


# Check the distribution of the feature applying np.sqrt()
plt.figure(figsize=(12,4))
sns.histplot(np.sqrt(df_version['TotalBsmtSF']),kde=True,bins=60)
# should convert to log or sqrt where dist is not normal??


# In[55]:


# Check the distribution of the feature
plt.figure(figsize=(12,4))
sns.histplot(df_version['GrLivArea'],kde=True,bins=60)


# In[56]:


# Check the distribution of the feature applying np.sqrt()
plt.figure(figsize=(12,4))
sns.histplot(np.sqrt(df_version['GrLivArea']),kde=True,bins=60)


# In[57]:


# Check the distribution of the feature
plt.figure(figsize=(12,4))
sns.histplot(df['1stFlrSF'],kde=True,bins=60)


# In[58]:


# Check the distribution of the feature applying np.sqrt()
plt.figure(figsize=(12,4))
sns.histplot(np.sqrt(df['1stFlrSF']),kde=True,bins=60)


# In[59]:


# Check the distribution of the feature
plt.figure(figsize=(12,4))
sns.histplot(df['GarageArea'],kde=True,bins=60)


# In[60]:


# Check the distribution of the feature applying np.sqrt()
plt.figure(figsize=(12,4))
sns.histplot(np.sqrt(df['GarageArea']),kde=True,bins=60)


# In[ ]:





# #### Outliers

# #### Delete outliers on train set

# In[61]:


# Method to find outliers base on standard deviation

def std_filter(df,feature): 
    std_feature = df[feature].std()    
    filter_output = df[abs(df[feature] - df[feature].mean()) >= 3*std_feature].index.tolist()
    return filter_output


# In[62]:


# Try the method on the copy df_version and using SalePrice outliers as an example since 
# it is a manipulation that could not be performednon the test dataset because it does 
# not have the SalePrice column
df_version['SalePrice'].mean()


# In[63]:


outliers_SalePrice =std_filter(df_version,'SalePrice')


# In[64]:


# df_version.loc[outliers_SalePrice,'SalePrice']


# In[65]:


# On the folowing features the outliers values will be deleted:
        # TotalBsmtSF            
        # GrLivArea


# In[66]:


# Check num of rows
df_numerical_final.shape


# In[67]:


# Check num of rows
df_categorical.shape


# In[68]:


# Applying the method to find outliers
outliers_TotalBsmtSF =std_filter(df_numerical_final,'TotalBsmtSF')


# In[69]:


# Applying the method to find outliers
outliers_GrLivArea =std_filter(df_numerical_final,'GrLivArea')


# In[70]:


len(outliers_TotalBsmtSF)


# In[71]:


len(outliers_GrLivArea)


# In[72]:


total_outliers = set(outliers_TotalBsmtSF + outliers_GrLivArea)


# In[73]:


len(total_outliers)


# In[ ]:





# In[74]:


df_numerical_final = df_numerical_final.drop(index=total_outliers)


# In[75]:


# Check num of rows
df_numerical_final.shape


# In[76]:


df_categorical = df_categorical.drop(index=total_outliers)


# In[77]:


# Check num of rows
df_categorical.shape


# #### Data distribution changes 

# In[78]:


# Transform the features using np.sqrt() to reduce the impact of extreme values
# and make the distribution more symmetrical

df_numerical_final['TotalBsmtSF'] = np.sqrt(df_numerical_final['TotalBsmtSF'])

df_numerical_final['GrLivArea'] = np.sqrt(df_numerical_final['GrLivArea'])


# In[ ]:





# #### One hot encoding on categorical features - train dataset

# In[79]:


# OneHotEncoder is a sklearn transformer that perform one hot encoding on categorical features
# The code import the packages needed
from sklearn.preprocessing import OneHotEncoder


# In[80]:


# the code below creates an instance of OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore' )


# In[ ]:





# In[81]:


# the code below stores in a variable the output of one hot encoding on categorical features
df_categorical_final = pd.DataFrame(onehot_encoder.fit_transform(df_categorical),
                           columns=onehot_encoder.get_feature_names(),
                          index=df_categorical.index)


# In[82]:


df_categorical_final.shape


# In[ ]:





# In[83]:


df_final = pd.concat([df_numerical_final,df_categorical_final],axis=1)


# In[84]:


df_final.head()


# In[85]:


# X = df_final


# In[86]:


# y = df['SalePrice']


# In[ ]:





# ### <div class="alert alert-block alert-info">
# 

# ### <div class="alert alert-block alert-info">

# ### Test dataset manipulation and filling missing values

# In[87]:


# the code below load the dataset
house_test = pd.read_csv('house_test.csv')


# In[88]:


# the code below creates a copy of the original dataset
dft = house_test.copy()


# In[89]:


# the code below inspects the dataset with .info() method
dft.info()


# In[90]:


# Removing the same features that were removed in the train dataset
dft.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[91]:


# The code below show the categorical features where are missing values
dft.select_dtypes(include='object').isnull().sum().sort_values(ascending=False).head(19)


# In[ ]:





# In[ ]:





# In[ ]:





# ### Missing values on categorical features - test dataset

# In[92]:


# According to the information in the database, the lack of data on some features 
# is due to the non-existence of the feature. Therefore the missing values on categorical features 
#can be filled for example with 'None'
# The code bellow fill the missing values with 'None' on the following features: MasVnrType,
# BsmtQual,BsmtCond,BsmtExposure,BsmtFinType1,
# BsmtFinType2,FireplaceQu,GarageType,GarageFinish,GarageQual,GarageCond,


# In[93]:


# Imputing missing values on categorical features
dft[['MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond']] = dft[['MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond']].fillna('None')


# #### Missing values on 'MSZoning'

# In[94]:


# Filling missing values on 'MSZoning' based on the mode


# In[95]:


dft['MSZoning'].value_counts()


# In[96]:


dft['MSZoning'] = dft['MSZoning'].fillna('RL')


# ####  Missing values on 'Utilities'

# In[97]:


# Filling missing values on 'Utilities' based on the mode


# In[98]:


dft['Utilities'].value_counts()


# In[99]:


dft['Utilities'] = dft['Utilities'].fillna('AllPub')


# #### Missing values on 'Functional'

# In[100]:


# Filling missing values based on the mode


# In[101]:


dft['Functional'].value_counts()


# In[102]:


dft['Functional'] = dft['Functional'].fillna('Typ') 


# #### Missing values on 'KitchenQual'

# In[103]:


# Filling missing values based on the mode considering the 'Neighborhood'


# In[104]:


dft[dft['KitchenQual'].isnull()]['Neighborhood']


# In[105]:


dft[dft['Neighborhood']=='ClearCr']['KitchenQual'].value_counts()


# In[106]:


dft['KitchenQual'] = dft['KitchenQual'].fillna('TA')


# #### Missing values on 'SaleType'

# In[107]:


# Filling missing values based on the mode


# In[108]:


dft['SaleType'].value_counts()


# In[109]:


dft['SaleType'] = dft['SaleType'].fillna('WD')


# #### Missing values on 'Exterior2nd'

# In[110]:


# Filling missing values based on the mode considering the 'Neighborhood'


# In[111]:


dft[dft['Exterior2nd'].isnull()]['Neighborhood']


# In[112]:


dft[dft['Neighborhood']=='Edwards']['Exterior2nd'].value_counts()


# In[113]:


dft['Exterior2nd'] = dft['Exterior2nd'].fillna('Wd Sdng')


# #### Missing values on 'Exterior1st'

# In[114]:


# Filling missing values based on the mode considering the 'Neighborhood'


# In[115]:


dft[dft['Exterior1st'].isnull()]['Neighborhood']


# In[116]:


dft[dft['Neighborhood']=='Edwards']['Exterior1st'].value_counts()


# In[117]:


dft['Exterior1st'] = dft['Exterior1st'].fillna('Wd Sdng')


# In[118]:


# dft.select_dtypes(include='object').isnull().sum().sort_values(ascending=False).head(19)


# ### Missing values on numerical features - test dataset

# In[119]:


# According to the information in the database, the lack of data on some features 
# is due to the non-existence of the feature. Therefore on thr numerical features the value 
# can be filled for example with (0.0)


# In[120]:


# Imputing missing values on numerical features
dft[['MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','GarageCars','GarageArea','BsmtFullBath','BsmtHalfBath']] = dft[['MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','GarageCars','GarageArea','BsmtFullBath','BsmtHalfBath']].fillna(0.0)


# #### Missing values on 'GarageYrBlt'

# In[ ]:





# In[121]:


# Filling missing values on the feature 'GarageYrBlt' based on the feature 'GarageCond'
dft[dft['GarageYrBlt'].isnull()]['GarageCond'].value_counts()


# In[122]:


dft['GarageYrBlt'] = dft['GarageYrBlt'].fillna(0.0)


# #### Missing numerical values on 'LotFrontage' predicted with IterativeImputer - test data

# In[123]:


# the code below stores in a variable the categorical features of the dataframe
dft_categorical = dft.select_dtypes(include='object')


# In[124]:


# the code below stores in a variable the numerical features of the dataframe
dft_numerical = dft.select_dtypes(exclude='object')


# In[125]:


# the code below stores in a variable the predicted missing values on numerical features
Xt_numerical = iterative_imputer.transform(dft_numerical)


# In[126]:


Xt_numerical


# In[127]:


# the code below creates a dataframe with the numerical features without missing values
dft_numerical_final = pd.DataFrame(Xt_numerical,columns=dft_numerical.columns,index=dft_numerical.index)


# In[128]:


dft_numerical_final.info()


# In[ ]:





# In[129]:


# The code below show the numerical features where are missing values
dft.select_dtypes(exclude='object').isnull().sum().sort_values(ascending=False).head(12)


# #### VERSION 1 (test set)

# In[ ]:





# In[130]:


# Changes proposed in version 1
# The following code removes numerical features that are highly correlated with each other

dft_numerical_final = dft_numerical_final.drop(high_collinearity,axis=1)


# In[131]:


# The following code removes numerical features that are low correlated with the label

dft_numerical_final = dft_numerical_final.drop(low_correlated_to_drop,axis=1)


# In[ ]:





# #### VERSION 2 (test set)

# ##### Replace outliers for the mean value instead of deleted because no row can be deleted as kaggle submission requirement expects 1460 rows to be loaded

# In[132]:


# Method to find outliers base on standard deviation and replace it for the mean value
def std_filter_2(df,feature): 
    std_feature = df[feature].std()    
    filter_output = df[abs(df[feature] - df[feature].mean()) >= 3*std_feature].index.tolist()
    df.loc[filter_output,feature] = df[feature].mean()
    
    return filter_output


# In[133]:


dft_numerical_final.shape


# In[134]:


# Check the mean of the feature
dft_numerical_final['TotalBsmtSF'].mean()


# In[135]:


# Check the mean of the feature
dft_numerical_final['GrLivArea'].mean()


# In[ ]:





# In[136]:


# Applying the method to replace outliers for the mean value
outliers_TotalBsmtSF_2 =std_filter_2(dft_numerical_final,'TotalBsmtSF')


# In[137]:


# Check that outliers have the new value mean of the feature
dft_numerical_final.loc[outliers_TotalBsmtSF_2,'TotalBsmtSF']


# In[138]:


# Applying the method to replace outliers for the mean value
outliers_GrLivArea_2 =std_filter_2(dft_numerical_final,'GrLivArea')


# In[139]:


# Check that outliers have the new value mean of the feature
dft_numerical_final.loc[outliers_GrLivArea_2,'GrLivArea']


# In[140]:


dft_numerical_final.shape


# In[141]:


dft_categorical.shape


# #### Data distribution changes

# In[142]:


# Transform the features using np.sqrt() to reduce the impact of extreme values
# and make the distribution more symmetrical

dft_numerical_final['TotalBsmtSF'] = np.sqrt(dft_numerical_final['TotalBsmtSF'])

dft_numerical_final['GrLivArea'] = np.sqrt(dft_numerical_final['GrLivArea'])


# In[ ]:





# #### OneHotEncoding on categorical features - test dataset

# In[143]:


# the code below stores in a variable the output of OneHotEncoding on categorical features
dft_categorical_final = pd.DataFrame(onehot_encoder.transform(dft_categorical),
                        columns=onehot_encoder.get_feature_names(),
                        index=dft_categorical.index)


# In[144]:


dft_categorical_final.shape


# In[145]:


# The code below concatenate the numerical features dataframe without missing values 
# with the categorical features dataframe transformed using OneHotEncoding
dft_final = pd.concat([dft_numerical_final,dft_categorical_final],axis=1)


# In[146]:


dft_final.head()


# In[147]:


dft_final.shape


# In[148]:


df_final.head()


# In[149]:


df_final.shape # including ['SalePrice']


# In[150]:


df_final['SalePrice']


# In[ ]:





# In[ ]:





# ### <div class="alert alert-block alert-info">

# ### <div class="alert alert-block alert-info">

# In[ ]:





# ### Models

# In[151]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# In[ ]:





# In[152]:


# The code below store the features on the variable X 
X = df_final.drop('SalePrice',axis=1)


# In[153]:


# The code below store the label on the variable y 
y = df_final['SalePrice']


# In[154]:


# The code below perform the split of the data into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[155]:


sum(X_train.isnull().any())


# In[156]:


# Check the shape of the splits to fit the models
print('X_train', X_train.shape)
print('X_test', X_test.shape)
print('y_train', y_train.shape)
print('y_test', y_test.shape)


# In[ ]:





# #### StandardScaler

# In[157]:


# Create an instance of StandardScaler
std = StandardScaler()


# In[158]:


# The code below scale the features
X_train_scaled = std.fit_transform(X_train)


# In[159]:


X_test_scaled = std.transform(X_test)


# #### LinearRegression

# In[160]:


# Only without scaling


# In[161]:


# The code below creates an instance of the model
linear_regression_1 = LinearRegression()


# In[162]:


# The code below fit the model without scaling X_train
linear_regression_1.fit(X_train,y_train)


# In[163]:


# The code below creates predictions without scaling X_test
pred_linear_regression_1 = linear_regression_1.predict(X_test)


# In[164]:


# The code below fit the model with the train scaled data to find parameters between features and label
# linear_regression_1.fit(X_train_scaled,y_train)


# In[165]:


# The code below creates predictions using test data based on the parameters learned 
# on the fit method
# pred_linear_regression_1 = linear_regression_1.predict(X_test_scaled)


# In[166]:


# Root Mean Squared Error (RMSE) between the truth and the predictions
RMSE_linear_V3 = np.sqrt(mean_squared_error(y_test,pred_linear_regression_1))


# In[167]:


RMSE_linear_V3


# In[168]:


np.sqrt(mean_squared_error(y_test,pred_linear_regression_1))/df['SalePrice'].mean()


# In[169]:


df['SalePrice'].describe()


# In[ ]:





# In[ ]:





# #### ElasticNet

# In[170]:


# Only scaled data, without scaling take to much time


# In[171]:


# The code below creates a grid to combine parameters
param_grid = {'alpha':[.3,.5,1,2],'l1_ratio':[.5,.65,.77,.87,.99]}


# In[172]:


elastic_net = ElasticNet(max_iter=1000000,random_state=101)


# In[173]:


grid_elastic_net = GridSearchCV(elastic_net, param_grid=param_grid, scoring='neg_mean_squared_error',cv=15,verbose=1)


# In[174]:


# Fit the model scaling X_train
grid_elastic_net.fit(X_train_scaled,y_train)


# In[175]:


# grid_elastic_net.fit(X_train,y_train)


# In[176]:


# Fit the model without scaling X_train
# grid_elastic_net.fit(X_train,y_train)


# In[177]:


grid_elastic_net.best_params_


# In[178]:


pred_grid = grid_elastic_net.predict(X_test_scaled)


# In[179]:


# The code below creates predictions based on X_test without scaling
# pred_grid = grid_elastic_net.predict(X_test)


# In[215]:


# Root Mean Squared Error (RMSE) between the truth and the predictions
RMSE_elastic_V3 = np.sqrt(mean_squared_error(y_test,pred_grid))


# In[181]:


np.sqrt(mean_squared_error(y_test,pred_grid))/df['SalePrice'].mean()


# In[216]:


RMSE_elastic_V3


# In[ ]:





# In[ ]:





# #### RandomForestRegressor

# In[183]:


# very similar result, use scaled data


# In[184]:


rf_model = RandomForestRegressor(random_state=101)


# In[244]:


param_grid1 = {'n_estimators':[100,150,200], # number of trees inside the classifier         
               'max_samples':[0.85,0.9],  # number of samples to draw from X
               'max_depth': [10,20,25], # the depth of the tree
              'max_features':[30,100,120]} # number of features


# In[186]:


grid_rf = GridSearchCV(rf_model,param_grid1)


# In[187]:


grid_rf.fit(X_train_scaled, y_train)


# In[188]:


# grid_rf.fit(X_train, y_train)


# In[189]:


grid_rf.best_params_


# In[190]:


grid_rf_prediction = grid_rf.predict(X_test_scaled)


# In[191]:


# grid_rf_prediction = grid_rf.predict(X_test)


# In[192]:


# Root Mean Squared Error (RMSE) between the truth and the predictions
RMSE_randomf_V3 = np.sqrt(mean_squared_error(y_test,grid_rf_prediction))


# In[193]:


RMSE_randomf_V3


# In[194]:


np.sqrt(mean_squared_error(y_test,grid_rf_prediction))/df['SalePrice'].mean()


# In[ ]:





# In[195]:


# RandomForestRegressor VERSION 0 rf_model.fit(X_train, y_train) very similar result, use scaled data.


# #### GradientBoostingRegressor

# In[196]:


# very similar result, use scaled data


# In[197]:


from sklearn.ensemble import GradientBoostingRegressor


# In[198]:


param_grid2 = {'n_estimators':[50,100,200],
             'learning_rate':[.03,.05,.1],
             'max_depth':[5,7,9],
               'subsample':[.6],
              'max_features':[15,30,90,130,220]}


# In[199]:


gbr_model = GradientBoostingRegressor(random_state=101)


# In[200]:


gbr_grid = GridSearchCV(gbr_model,param_grid2)


# In[201]:


gbr_grid.fit(X_train_scaled,y_train)


# In[202]:


gbr_grid.best_params_


# In[203]:


gbr_prediction = gbr_grid.predict(X_test_scaled)


# In[204]:


# Root Mean Squared Error (RMSE) between the truth and the predictions
RMSE_gradient_V3 = np.sqrt(mean_squared_error(y_test,gbr_prediction))


# In[205]:


RMSE_gradient_V3


# In[206]:


# Ratio (RMSE)/df['SalePrice'].mean()
np.sqrt(mean_squared_error(y_test,gbr_prediction))/df['SalePrice'].mean()


# In[207]:


# RMSE_gradient_V0 RMSE_randomf_V0 RMSE_elastic_V0 RMSE_linear_V0


# In[217]:


result_table = {'Linear Regression':[RMSE_linear_V3],'Elastic Net':[RMSE_elastic_V3],'Random Forest':[RMSE_randomf_V3],'Gradient Boosting':[RMSE_gradient_V3]}


# In[220]:


result_table1 = pd.DataFrame(data=result_table,index = ['Version 3'])


# In[221]:



result_table1


# ### <div class="alert alert-block alert-info">

# ### <div class="alert alert-block alert-info">

# #### Submission to Kaggle

# In[ ]:





# In[228]:


# Check the shape of the test set
dft_final.shape


# In[229]:


df_final.shape


# In[230]:


# Check the shape of the train set
X.shape


# In[ ]:


# std = StandardScaler()


# In[238]:


# The code below scale the features on the test set with the same scaler used on the train set
dft_final_scaled = std.transform(dft_final)


# In[ ]:


# The code below calculates predictions based on the parameters learned fitting the train 
# data to the estimator
# kaggle_prediction = linear_regression_1.predict(dft_final_scaled) # LinearRegression


# In[233]:


# kaggle_prediction = grid_elastic_net.predict(dft_final_scaled) # ElasticNet


# In[239]:


# kaggle_prediction = grid_rf.predict(dft_final_scaled) # RandomForestRegressor


# In[225]:


# kaggle_prediction = gbr_grid.predict(dft_final_scaled) # GradientBoosting


# In[240]:


# submission = pd.DataFrame({ 'Id': house_test['Id'].values, 'SalePrice': kaggle_prediction })


# In[241]:


# submission.head()


# In[242]:


# submission['SalePrice'] = submission['SalePrice'].astype(int)


# In[243]:


# submission.to_csv('subm/HOUSE_VERSION_3_rf.csv',index=False)


# In[ ]:




