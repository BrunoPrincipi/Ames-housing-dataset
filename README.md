# House Prices - Advanced Regression Techniques

## Background
For this project, I am using the [Ames Housing dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview).
The Ames Housing dataset was compiled by Dean De Cock for use in data science education. It's an incredible alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset. 

## My approach
From the downloaded kaggle database, I have created 4 versions where different data cleaning and preprocessing tasks are applied. Then each version of the database is evaluated using different algorithms combined with a grid of hyperparameters. Finally, the best training results were tested by submitting it through Kaggle.
The best results both in the training stage and in the evaluation of Kaggle were obtained using the gradient boosting regressor algorithm.

Gradient boosting:
“Another very popular boosting algorithm is Gradient Boosting. Just like AdaBoost, Gradient Boosting works by sequentially adding predictors to an ensemble, each one correcting its predecessor. However, instead of tweaking the instance weights at every iteration like AdaBoost does, this method tries to fit the new predictor to the residual errors made by the previous predictor”. 

For more details, see Aurélien Géron., "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow", third edition (2022): 226–230.


Goals of this project:

Predict the sale price of the houses based on the given features.

Learn and experiment how different data cleaning/preprocessing tasks affect model predictions.

Learn and experiment how the different hyperparameters affect model predictions.

## Files inside this repo
This Project includes:
  4 executable files, one for each version of the dataset.
  1 word file with information about the dataset, cleaning process, results and conclusions.

To run the executable files will be needed to download the train and test dataset (train.csv and test.csv)
Link to dowload the train and test dataset:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data


## To DO

Implement a basic neural network.

Improve the neural network

# Contact Details
