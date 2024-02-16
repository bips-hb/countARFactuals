predict.xgboost = function(object, newdata){
  newData_x = xgb.DMatrix(data.matrix(newdata), missing = NA)
  results = predict(object, newData_x, reshape = TRUE)
  return(results)
}
