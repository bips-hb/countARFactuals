library("xgboost")
library("iml")
library("data.table")

get_data_model_python = function(datanam, x_interest_ids, model_paras) {
  
  dir.create("python/synthetic/data", showWarnings = FALSE)
  dir.create("python/synthetic/xgboost_paras", showWarnings = FALSE)
  dir.create("python/synthetic/x_interests", showWarnings = FALSE)
  
  data_path = file.path("python/synthetic", paste0(datanam, ".csv"))
  
  # read data
  dt = data.table(read.csv(data_path, header = TRUE, sep = ","))
  if ("log_prob" %in% names(dt)) dt[, log_prob := NULL]
  
  if ("True" %in% unique(dt$y)) {
    dt$y = ifelse(dt$y == "True", 1, 0)
  }
  
  # cat("Feature names", names(dt))
  
  # x_interest
  x_interest_ids = sample(seq_len(nrow(dt)), size = 10L)
  x_interest = dt[x_interest_ids, ]
  dt = dt[-x_interest_ids,]
  
  # set up predictor object
  ## TODO: model_paras should be imported from python
  ## https://stackoverflow.com/questions/61663184/porting-xgboost-model-between-python-and-r
  dtrain <- xgboost::xgb.DMatrix(
    data = as.matrix(dt[,.SD, .SDcols = !c('y')]),
    label = dt$y 
  )
  
  params <- list(
    max_depth = 3
  )
  
  xgbmodel <- xgboost::xgb.train(
    params = list(),
    dtrain,
    nrounds = 10
  )
  
  # save infos
  xgboost::xgb.save(xgbmodel, file.path("python/synthetic/xgboost_paras", paste0(datanam, ".model")))
  write.csv(x_interest, file = file.path("python/synthetic/x_interests", paste0(datanam, ".csv")))
  write.csv(dt, file = file.path("python/synthetic/data/", paste0(datanam, ".csv")))
  
  
  # test if model works with x_interest
  predictor = Predictor$new(model = xgbmodel, data = dt, y = "y", predict.function = predict.xgboost)
  pred =  predictor$predict(x_interest)
  # print(head(pred))
  # if (ncol(pred) > 1) {
  #   current_class = apply(pred, 1, function(pr) names(which.max(pr)))
  # } else {
  #   current_class = apply(pred, 1, function(pr) ifelse(pr > 0.5, 1, 0))
  # }
  
  saveRDS(predictor, file = file.path("R_experiments/predictors", paste0(datanam, ".rds")))
  return(invisible())
}


predict.xgboost = function(object, newdata){
  newData_x = xgb.DMatrix(data.matrix(newdata), missing = NA)
  results = predict(object, newData_x)
  return(results)
}


if (FALSE) {
  pawel = get_data_model_python(datanam = "pawelczyk")
  cassini = get_data_model_python(datanam = "cassini")
  two_sines = get_data_model_python(datanam = "two_sines")
  bn = get_data_model_python(datanam = "bn_1")
}
