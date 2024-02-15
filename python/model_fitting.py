import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import scipy.stats as stats


data_name = 'bn_1'
loadpath = 'python/synthetic/'
train_size = 5000
x_interest_size = 50

df = pd.read_csv(loadpath + '{}.csv'.format(data_name), index_col=0)
# Convert columns to integer if possible
df = df.apply(pd.to_numeric, errors='ignore')

assert df.shape[0] >= train_size + x_interest_size

## Split the data into training and testing sets

df_train = df.iloc[:train_size]
df_rest = df.iloc[train_size:]
df_interest = df_rest.sample(x_interest_size)

X_train, y_train = df_train.drop('y', axis=1), df_train['y']
X_rest, y_rest = df_rest.drop('y', axis=1), df_rest['y']


## hyperparameter tuning

# Define the hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'subsample': [0.5, 0.7, 1]
}

# Create the XGBoost model object
xgb_model = xgb.XGBClassifier()

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

model = xgb.train(
    params,
    dtrain
)

y_proba_train = model.predict(dtrain)
y_proba_rest = model.predict(drest)

print(roc_auc_score(y_train, y_proba_train))
print(roc_auc_score(y_rest, y_proba_rest))


df_train.to_csv(loadpath + 'data/{}.csv'.format(data_name), index=False)
df_interest.to_csv(loadpath + 'x_interests/{}.csv'.format(data_name), index=False)
model.save_model(loadpath + 'xgboost_paras/{}.model'.format(data_name))

