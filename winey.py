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

# Output data
print(data.head())
print(data.shape)
print(data.describe())

# Seperating the target feature data
# from the rest of the data
y = data.quality 
X = data.drop('quality', axis=1)

# Splitting data into training and test data sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

# Data preprocessing...
pipeline = make_pipeline(preprocessing.StandardScaler(), 
                        RandomForestRegressor(n_estimators=100))

# Parameters to adjust for Random Forest Regressor Model
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

# Performing cross-validation with pipeline on model
rf_clf = GridSearchCV(pipeline, hyperparameters, cv=10)

rf_clf.fit(X_train, y_train)

# Evalutating effectiveness of model pipeline
# on the test data
pred = rf_clf.predict(X_test)

print(r2_score(y_test, pred))
print(mean_squared_error(y_test, pred))

# Dump the random forest regressor model
# into a file
joblib.dump(rf_clf, 'rf_regressor.pkl')