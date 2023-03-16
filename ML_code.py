import multiprocessing
from functools import partial
from multiprocessing import Pool
from multiprocessing import cpu_count
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

class data_proc:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,df_in,features,target,rnd_state=42):
        self.data = df_in
        self.y = df_in[target].copy()
        self.X = df_in[features]
        self.features = features
        self.rnd_state = rnd_state
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(self.X, self.y)
        self.n_jobs = cpu_count() # <--- CPUs available for multithreading
        self.colors = ['#ebac23','#b80058','#008cf9','#006e00','#00bbad',
                       '#d163e6','#b24502','#ff9287','#5954d6','#00c6f8',
                       '#878500','#00a76c','#bdbdbd','#000000','#b80058',
                       '#ebac23']
    def group_split(self, group, train_size=0.70):
        splitter = GroupShuffleSplit(train_size=train_size)
        train, test = next(splitter.split(self.X,self.y,groups=group))
        self.train_X = self.X.iloc[train]
        self.test_X = self.X.iloc[test]
        self.train_y = self.y.iloc[train]
        self.test_y = self.y.iloc[test]
        return (self.train_X,self.test_X,self.train_y,self.test_y)
    def get_mae(self,max_leaf_nodes):
        model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=self.rnd_state)
        model.fit(self.train_X, self.train_y)
        preds_val = model.predict(self.val_X)
        mae = mean_absolute_error(self.val_y, preds_val)
        return(mae)
    def get_model(self,max_leaf_nodes=100,w_resids=False,model_type='RF'):
        if w_resids == False:
            train_X_data = self.train_X.drop('resids', axis=1)
            val_X_data = self.val_X.drop('resids', axis=1)
            # print('Trained on: ')
            # print(train_X_data.columns.values)
        else:
            train_X_data = self.train_X.copy()
            val_X_data = self.val_X.copy()
        if model_type == 'RF':
            self.model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes,random_state=self.rnd_state)
        elif model_type == 'DT':
            self.model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=self.rnd_state)
        else:
            print('model_type should be either RF or DT. Here comes the error...')
        self.model.fit(train_X_data,self.train_y)
        val_pred = self.model.predict(val_X_data)
        val_mae = mean_absolute_error(val_pred,self.val_y)
        return val_pred,val_mae