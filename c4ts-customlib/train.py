
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from azureml.core.run import Run
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

from utils import *
    
os.makedirs('./outputs', exist_ok=True)

# Data Preparation
df = pd.read_csv('AssetData_Historical.csv')
df.drop(['Machine_ID', 'District'], axis=1, inplace=True)
df['Latitude'] = cleanLongLat(df['Latitude'])
df['Longitude'] = cleanLongLat(df['Longitude'])
X = df.drop('Failure_NextHour', 1)
y = df['Failure_NextHour']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, stratify = y)

X_prep, s, pca= prepare(X_train, fit = True)

run = Run.get_context()

param_grid = {'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
              'n_estimators': [100, 200, 300, 400, 500]
}

model = GradientBoostingClassifier(loss = 'exponential')
kf = StratifiedKFold(n_splits = 5, shuffle = True)
gridsearch = GridSearchCV(model, param_grid, 
                          scoring = 'f1_weighted',
                          n_jobs = -1,
                          cv = kf)
weights = y_train * 3 + 1
result = gridsearch.fit(X_prep, y_train, sample_weight = weights)

run.log('bestScore', result.best_score_)
run.log('bestParam', result.best_params_)
run.log('valMean', result.cv_results_['mean_test_score'])
run.log('valStd', result.cv_results_['std_test_score'])
run.log('valParams', result.cv_results_['params'])
run.log('FeatureImportance', result.feature_importances_)

#################
#Fit Final Model#
#################

X, s, pca = prepare(X, fit = True)
best_model = result.estimator.fit(X, y, sample_weight = y*4 + 1)

pickle.dump(s, open('./outputs/scaler.pkl', 'wb'))
pickle.dump(pca,open('./outputs/pca_transform.pkl','wb'))

import time
model_name = 'GBT_{}'.format(time.time())
with open(model_name, 'wb') as f:
    joblib.dump(value = best_model, filename = './outputs/{}.pkl'.format(model_name))
