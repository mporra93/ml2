#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mlflow
from mlflow.client import MlflowClient

import os
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy  as np
from numpy import random as rd
from matplotlib import pyplot as plt


from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

import pandas as pd


#This will be used in the next class
os.chdir('/home/ml2/') 


## This db could be an external postgres database
mlflow.set_tracking_uri('sqlite:///mlruns.db')

## This will fail in databricks because the experiment_id is a random hash

new_experiment_id = 0  
list_mlflow_experiments =  mlflow.search_experiments()
if len(list_mlflow_experiments):
    list_experiment_id = list(map(lambda list_mlflow_experiments: int(list_mlflow_experiments.experiment_id), list_mlflow_experiments ))
    last_experiment_id =  max(list_experiment_id)
    new_experiment_id  = last_experiment_id + 1
    
    mlflow.create_experiment(str(new_experiment_id))


# This should be a query from a database
DATASET_PATH = '/home/ml2/data_playlist.csv'
df = pd.read_csv(DATASET_PATH, delimiter=',')
X = df.drop(['label'],axis = 1)
y = df['label']

# preparo el dataset
#creamos mapping para TEMPO, para transformarlo en categorica y mappeamos a binarios
tempo_mappings = {
    (40,60)   : '000', #'lento',
    (60,66)   : '001', #'Larghetto',
    (66,76)   : '010', #'Adagio',
    (76,108)  : '011', #'Andante',
    (108,120) : '100', #'Moderato',
    (120,168) : '101', #'Allegro',
    (168,200) : '110', #'Presto',
    (200,216) : '111', #'Prestissimo',
                }


def map_tempos(x):
    for key in tempo_mappings:
        if x >= key[0] and x <= key[1]:
            return tempo_mappings[key]

df['tempo'] = df['tempo'].apply(map_tempos)

#lo mismo para liveness
df['liveness'] = df['liveness'].apply(lambda d: 1 if d>0.8 else 0)

mlflow.sklearn.autolog(max_tuning_runs = None)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=2022)
 
 
### POR AHORA VAMOS A USAR EL DATASET DE TEST HASTA SOLUCIONAR EL PROBLEMA DEL NUESTRO

# # This should be a query from a database
# X = sklearn.datasets.load_iris().data
# y = sklearn.datasets.load_iris().target

# # # Usando el autolog muchas metricas se guardan solas...
# mlflow.sklearn.autolog(max_tuning_runs = None)
# X_train, X_test, y_train, y_test = train_test_split(X,y)


def log_model(model,
              developer = None,
              experiment_id = None,
              grid = False,
              **kwargs):
    
    
    assert developer     is not None, 'You must define a developer first'
    assert experiment_id is not None, 'You must define a experiment_id first'
    
    
    with mlflow.start_run(experiment_id = experiment_id):

        mlflow.set_tag('developer',developer)
        
        #The default is to train just one model
        model = model(**kwargs)
        if grid:
            model = GridSearchCV(model,param_grid = kwargs)
        
        
        
        model.fit(X_train, y_train)
        test_acc = (model.predict(X_test) == y_test).mean()

        mlflow.log_metric('test_acc',test_acc)


#normal logging
log_model(RandomForestClassifier,'Matias', experiment_id = new_experiment_id)


