# House Prices - Advanced Regression Techniques

## Background
For this project, I am using the [Ames Housing dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview).
The Ames Housing dataset was compiled by Dean De Cock for use in data science education. It's an incredible alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset. 

## My approach
From the downloaded kaggle database, I have created 4 versions where different data cleaning and preprocessing tasks are applied. Then each version of the database is evaluated using different algorithms combined with a grid of hyperparameters. Finally, the best training results were tested by submitting it through Kaggle.
The best results both in the training stage and in the evaluation of Kaggle were obtained using the gradient boosting regressor algorithm.
![image](https://user-images.githubusercontent.com/125404145/227923257-af489251-583f-44d3-a938-e15ccae673a9.png)








Gradient boosting:
“Another very popular boosting algorithm is Gradient Boosting. Just like AdaBoost, Gradient Boosting works by sequentially adding predictors to an ensemble, each one correcting its predecessor. However, instead of tweaking the instance weights at every iteration like AdaBoost does, this method tries to fit the new predictor to the residual errors made by the previous predictor”. 

For more details, see Aurélien Géron., "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow", third edition (2022): 226–230.
Also check the scikitlearn link: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

Goals of this project:

Learn and experiment how different data cleaning/preprocessing tasks affect model predictions.

Learn and experiment how the different hyperparameters affect model predictions.

Predict the sale price of the houses based on the given features.

## Files inside this repo
This Project includes:
  Four executable files, one for each version of the dataset.
  A documentation as word file with information about the dataset, cleaning process, results and conclusions.

To run the executable files will be needed to download the train and test dataset (train.csv and test.csv)
Link to dowload the train and test dataset:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data


## To DO

Implement a basic neural network.

Improve the neural network

# Contact Details
Email: principi.bruno@gmail.com
