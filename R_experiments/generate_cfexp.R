library(devtools)
library(iml)
load_all("counterfactuals_R_package/")

library(fdm2id)
library(mlbench)
library(xgboost)
library(ggplot2)
library(ranger)
library(ggsci)
library(doParallel)
registerDoParallel(8L)
set.seed(1, "L'Ecuyer-CMRG")


generate_cfexp = function(datanam, x_interest_id = 1L, method = "CountARF", nondom = FALSE, subset_valid = TRUE) {
  
  # get model & data
  xgbmodel =  xgboost::xgb.load(file.path("python/synthetic/xgboost_paras", paste0(datanam, ".model")))
  x_interest = read.csv(file.path("python/synthetic/x_interests", paste0(datanam, ".csv")))[x_interest_id,]
  dt = read.csv(file.path("python/synthetic/data/", paste0(datanam, ".csv")), header = TRUE)
  
  # get desired class
  predictor = Predictor$new(model = xgbmodel, data = dt, y = "y", predict.function = predict.xgboost)
  pred = predictor$predict(x_interest)
  predictor$task = "classification"
  # if (ncol(pred) > 1) {
  #   target_class = apply(pred, 1, function(pr) names(which.min(pr)))
  # } else {
  #   target_class = as.character(apply(pred, 1, function(pr) ifelse(pr > 0.5, 0, 1)))
  # }
  target_class = "pred"
    if (pred < 0.5) {
      desired_prob = c(0.5, 1)
    } else {
      desired_prob = c(0, 0.5)
    }

  
  if (method == "MOC") {
    cac = MOCClassif$new(predictor = predictor, n_generations = 50L)
  } else {
    cac = CountARFactualClassif$new(predictor = predictor)
  }
  cfexpobj = cac$find_counterfactuals(
    x_interest, desired_class = target_class, desired_prob = desired_prob
  )
  
  if (subset_valid) {
    cfexpobj$subset_to_valid()
  }
  
  if (nondom) {
    cfexp = cfexpobj$evaluate(arf = cac$.__enclos_env__$private$arf)
    fitnesses = cfexp[, c("dist_x_interest", "no_changed", "neg_lik", "dist_target")]
    cfexp = cfexp[miesmuschel::rank_nondominated(-as.matrix(fitnesses))$fronts == 1, 1:2]
  } else {
    cfexp = cfexpobj$data
  }
  cfexp$y = NA
  
  # Put it all together, export
  df_orig = data.table(Data = "Train", dt)
  df_orig = rbind(df_orig, data.table(Data = "x_interest", x_interest))
  df_orig = rbind(df_orig, data.table(cfexp, Data = "Counterfactuals"))[, Dataset := datanam][, Method := method]
  
  # Replace Class by predicted Class
  preds = predictor$predict(df_orig)
  df_orig$y = ifelse(preds > 0.5, 1, 0)
  
  return(df_orig)
}


predict.xgboost = function(object, newdata){
  newData_x = xgb.DMatrix(data.matrix(newdata), missing = NA)
  results = predict(object, newData_x, reshape = TRUE)
  return(results)
}



if (FALSE) {
  generate_cfexp(datanam = "cassini", x_interest_id = 1L)
  generate_cfexp(datanam = "pawelczyk", x_interest_id = 1L)
  generate_cfexp(datanam = "two_sines", x_interest_id = 1L)
  generate_cfexp(datanam = "bn_1", x_interest_id = 1L)
}


# Execute in parallel
dsets = c("pawelczyk", "cassini", "two_sines")
df_arf = foreach(d = dsets, .combine = rbind) %dopar% generate_cfexp(d, 1L, method = "CountARF")
df_moc = foreach(d = dsets, .combine = rbind) %dopar% generate_cfexp(d, 1L, method = "MOC")


# Set scales free but fix x-axis ticks
df = rbind(df_arf, df_moc)
df[Data == "Train", Data := paste0("Train_", y)]


# Scatter plot
ggplot(df, aes(x = x1, y = x2, color = Data)) + 
  geom_point(alpha = 0.75) + 
  scale_color_ordinal() + 
  facet_grid(Method ~ Dataset) + 
  theme_bw() + 
  theme(text = element_text(size = 14))

# MOC & ARF
df_arf_1 = generate_cfexp(datanam = "two_sines", x_interest_id = 1L)
df_moc_1 = generate_cfexp(datanam = "two_sines", method = "MOC", x_interest_id = 1L)

df = rbind(df_arf_1, df_moc_1)
df[Data == "Train", Data := paste0("Train_", y)]

ggplot(df, aes(x = x1, y = x2, color = Data)) + 
  geom_point(alpha = 0.75) + 
  scale_color_ordinal() + 
  facet_grid( ~ Method) + 
  theme_bw() + 
  theme(text = element_text(size = 14))

