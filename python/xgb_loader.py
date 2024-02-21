import xgboost as xgb
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np

dataname = 'bn_5'
loadpath = 'python/synthetic/xgboost_paras/{}.model'.format(dataname)

df_train = pd.read_csv('python/synthetic/data/{}.csv'.format(dataname))
df_interest = pd.read_csv('python/synthetic/x_interests/{}.csv'.format(dataname))

X_interest, y_interest = df_interest.drop('y', axis=1), df_interest['y']
X_train, y_train = df_train.drop('y', axis=1), df_train['y']

d_interest = xgb.DMatrix(X_interest, y_interest)
d_train = xgb.DMatrix(X_train, y_train)

bst = xgb.Booster()
bst.load_model(loadpath)

y_proba_interest = bst.predict(d_train)
print(y_proba_interest)

params = {
    'learning_rate': 0.1,
    'max_depth': 3,
    'objective': 'binary:logistic',
}

model = xgb.train(
    params,
    d_train
)

y_proba_interest = model.predict(d_interest)
y_proba_interest

model.save_model('test.model')

model_reload = xgb.Booster()
model_reload.load_model('test.model')

y_proba_interest = model_reload.predict(d_interest)
y_proba_interest