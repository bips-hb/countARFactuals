library(xgboost)

df = read.csv('python/synthetic/data/bn_1.csv')
model <- xgboost::xgb.load('python/synthetic/xgboost_paras/bn_1.model')

X_train <- df[,1:4]
y_train <- df[,5]

dtrain <- xgboost::xgb.DMatrix(
    data = as.matrix(X_train),
    label = y_train
)

y_proba_train <- predict(model, dtrain)