import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import scipy.stats as stats
import argparse
import numpy as np
from visualize import plot_pairs, get_savepath


names = ['bn_5', 'bn_100', 'bn_10', 'bn_50', 'cassini', 'pawelczyk', 'two_sines', 'bn_20', 'bn_10_v2', 'bn_5_v2', 'bn_50_v2']

# loadpath = 'python/synthetic/'
# data_name = 'bn_5'
# train_size = 5000
# x_interest_size = 50

parser = argparse.ArgumentParser()
parser.add_argument('loadpath')
parser.add_argument('name')
parser.add_argument('--train_size', type=int, default=5000)
parser.add_argument('--x_interest_size', type=int, default=50)
args = parser.parse_args()

assert args.name in names
data_name = args.name
loadpath = args.loadpath

train_size = args.train_size
x_interest_size = args.x_interest_size

if 'bn_' in data_name:
    df = pd.read_csv(loadpath + '{}.csv'.format(data_name))
else:
    df = pd.read_csv(loadpath + '{}.csv'.format(data_name))

# Convert columns to integer if possible
df['y'] = (df['y'] >= 1).astype(int)

assert df.shape[0] >= train_size + x_interest_size

## Split the data into training and test set

df_train = df.iloc[:train_size]
df_rest = df.iloc[train_size:2*train_size]
df_rest_filtered = df_rest[df_rest['y'] == 0]

def sample_interest(x_interest_size, candidates, background):
    df_interest = candidates.sample(x_interest_size)
    foreground = df_interest.copy()
    foreground['y'] = 'cf'
    df_visualize = pd.concat([background, foreground])
    plot_pairs(df_visualize, data_name, sample=False)
    if input('Happy with CFs (y/n)') == 'y':
        plot_pairs(df_visualize, data_name, sample=False, savepath=get_savepath(loadpath), show_plot=False)
        return df_interest
    else:
        return sample_interest(x_interest_size, candidates, background)


df_interest = sample_interest(x_interest_size, df_rest_filtered, df_rest.iloc[:200])

X_train, y_train = df_train.drop('y', axis=1), df_train['y']
X_rest, y_rest = df_rest.drop('y', axis=1), df_rest['y']
X_interest, y_interest = df_interest.drop('y', axis=1), df_interest['y']


## hyperparameter tuning

# Define the hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'subsample': [0.5, 0.7, 1]
}

# param_grid = {
#         'min_child_weight': [1, 5, 10],
#         'gamma': [0.5, 1, 1.5, 2, 5],
#         'subsample': [0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0],
#         'max_depth': [3, 4, 5]
# }

# Create the XGBoost model object
xgb_model = xgb.XGBClassifier(n_estimators=100, objective='binary:logistic', use_label_encoder=False)

# Create the GridSearchCV object
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best set of hyperparameters and the corresponding score
print("Best set of hyperparameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


## fit model with best hyperparameters on whole training data
params = grid_search.best_params_

dtrain = xgb.DMatrix(X_train, y_train)
drest = xgb.DMatrix(X_rest, y_rest)
dinterest = xgb.DMatrix(X_interest, y_interest)

model = xgb.train(
    params,
    dtrain
)

y_proba_train = model.predict(dtrain)
y_proba_rest = model.predict(drest)
y_proba_interest = model.predict(dinterest)

interest_candidates = np.sum(y_proba_interest < 0.5)
print(interest_candidates, 'negative predictions in x_interest out of ', len(y_proba_interest))

mean_class = np.mean(y_proba_rest >= 0.5)
print('proportion predicted 1s in test', mean_class)
assert 0.25 < mean_class < 0.75
assert interest_candidates >= 10

df_train.to_csv(loadpath + 'data/{}.csv'.format(data_name), index=False)
df_interest.to_csv(loadpath + 'x_interests/{}.csv'.format(data_name), index=False)
model.save_model(loadpath + 'xgboost_paras/{}.model'.format(data_name))

