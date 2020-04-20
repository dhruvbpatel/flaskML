import pandas as pd
import numpy as np
import pickle

#import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

path ='/data/iris.csv'


colname = ['sepal_length','sepal_width','petal_length','petal_width','species']


data = pd.read_csv(path,names=colname)

data = data.values

X = data[:,0:4]
y = data[:,4]


X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3,random_state=42)


rf = RandomForestClassifier(random_state=123,n_estimators=100,max_depth = 4)
rf.fit(X_train,y_train)

#xgb_cl = xgb.XGBClassifier()

rf_model= open('RFModel.pckl', 'wb')
pickle.dump(rf, rf_model)
rf_model.close()
