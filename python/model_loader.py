import xgboost as xgb
import pandas as pd


loadpath = 'python/synthetic/xgboost_paras/pawelczyk.model'

bst = xgb.Booster()
bst.load_model(loadpath)

df_train = pd.read_csv('python/synthetic/data/cassini.csv')
df_interest = pd.read_csv('python/synthetic/x_interests/cassini.csv')

X_interest, y_interest = df_interest.drop('y', axis=1), df_interest['y']
d_interest = xgb.DMatrix(X_interest, y_interest)

y_proba_test = bst.predict(d_interest)