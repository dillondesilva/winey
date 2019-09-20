# Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

# Importing Wine Data Set
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=";")

print(data.describe())

# Seperating the target data from training features
y = data.quality 
X = data.drop('quality', axis=1)

# Splitting data into train and test sets

# Train sets: Sets of data which are used
# to develop an ML model using a classifier of sorts

# Test sets: Sets of data which are used to
# test our trained data against

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

print(X_train_scaled.mean(axis=0))