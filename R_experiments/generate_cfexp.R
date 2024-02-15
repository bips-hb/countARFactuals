library(devtools)
library(iml)
load_all("counterfactuals_R_package/")

library(fdm2id)
library(mlbench)
library(xgboost)
library(ggplot2)
library(ranger)
library(ggsci)
library(doMC)
registerDoMC(8)
set.seed(1, "L'Ecuyer-CMRG")


generate_cfexp = function(datanam, x_interest_id = 1L, method = "CountARF", nondom = FALSE, subset_valid = TRUE) {
  
  # get model & data
  predictor = readRDS(file.path("R_experiments/predictors", paste0(datanam, ".rds")))
  x_interest = read.csv(file.path("python/synthetic/x_interests", paste0(datanam, ".csv")))[x_interest_id,]
  dt = read.csv(file.path("python/synthetic/data/", paste0(datanam, ".csv")), header = TRUE)
  
  # get desired class
  pred = predictor$predict(x_interest)
  if (ncol(pred) > 1) {
    target_class = apply(pred, 1, function(pr) names(which.min(pr)))
  } else {
    target_class = apply(pred, 1, function(pr) ifelse(pr > 0.5, 0, 1))
  }

  
  if (method == "MOC") {
    cac = MOCClassif$new(predictor = predictor, n_generations = 50L)
  } else {
    cac = CountARFactualClassif$new(predictor = predictor)
  }
  cfexpobj = cac$find_counterfactuals(
    x_interest, desired_class = target_class, desired_prob = c(0.5, 1)
  )
  
  if (nondom) {
    cfexp = cfexpobj$evaluate()
    fitnesses = cfexp[, c("dist_x_interest", "no_changed", "dist_train", "dist_target")]
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
  df_orig$y = colnames(preds)[apply(preds,1,which.max)]
  
  # subset valid
  if (subset_valid) {
    df_orig[ !(Data == "Counterfactuals" & y != target_class),] 
  } else {
    df_orig
  }
}


if (FALSE) {
  generate_cfexp(datanam = "cassini", x_interest_id = 1L)
  generate_cfexp(datanam = "pawelczyk", x_interest_id = 1L)
  generate_cfexp(datanam = "two_sines", x_interest_id = 1L)
  generate_cfexp(datanam = "bn_1", x_interest_id = 1L)
}


# Execute in parallel
dsets = c("pawelczyk", "cassini", "two_sines")
df_arf = foreach(d = dsets, .combine = rbind) %dopar% generate_cfexp(d, x_interest_id = 1L, method = "CountARF")
df_moc = foreach(d = dsets, idx = 1:10, .combine = rbind) %dopar% generate_cfexp(d, idx, method = "MOC")


# Set scales free but fix x-axis ticks
df = rbind(df_arf, df_moc)
df[Data == "Train", Data := paste0("Train_", y)]


# Scatter plot
ggplot(df, aes(x = X, y = Y, color = Data)) + 
  geom_point(alpha = 0.75) + 
  scale_color_ordinal() + 
  facet_grid(Method ~ Dataset) + 
  theme_bw() + 
  theme(text = element_text(size = 14))

