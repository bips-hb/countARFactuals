#### Simple examples
## based on https://github.com/bips-hb/arf_paper/blob/master/simulation/5.1_visual_patterns.R

# Load libraries, register cores, set seed
library(devtools)
load_all("counterfactuals_R_package/")
library(data.table)
library(arf)
library(fdm2id)
library(mlbench)
library(ggplot2)
library(ranger)
library(ggsci)
library(doMC)
registerDoMC(8)
set.seed(1, "L'Ecuyer-CMRG")

# Simulation function
sim_fun = function(n_trn, dataset, method = "CountARF", nondom = FALSE, subset_valid = TRUE) {
  # Simulate data
  if (dataset == 'twomoons') {
    x = data.twomoons(n = n_trn/2, graph = FALSE)
    x$Class = as.factor(gsub('Class ', '', x$Class))
  } else {
    if (dataset == 'cassini') {
      tmp = mlbench.cassini(n_trn)
    } else if (dataset == 'smiley') {
      tmp = mlbench.smiley(n_trn)
    } else if (dataset == 'shapes') {
      tmp = mlbench.shapes(n_trn)
    }
    x = data.frame(tmp$x, as.factor(tmp$classes))
    colnames(x) = c('X', 'Y', 'Class')
  }

  # Randomly select a point of interes
  idx = sample(1:n_trn, size = 1L)
  x_interest = x[idx, ]
  x = x[-idx,]
  
  # Fit model without obs of interest
  rf = ranger(Class ~ ., x, probability = TRUE)
  predictor = Predictor$new(model = rf, data = x, y = "Class")
  
  # get desired prob
  pred = predictor$predict(x_interest)
  target_class = names(which.min(pred))
  
  # Fit countARF
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
  cfexp$Class = NA
  
  # Put it all together, export
  df_orig = data.table(Data = "Train", x)
  df_orig = rbind(df_orig, data.table(Data = "x_interest", x_interest))
  df_orig = rbind(df_orig, data.table(cfexp, Data = "Counterfactuals"))[, Dataset := dataset][, Method := method]
  
  # Replace Class by predicted Class
  preds = predictor$predict(df_orig)
  df_orig$Class = colnames(preds)[apply(preds,1,which.max)]
  
  # subset valid
  if (subset_valid) {
    df_orig[ !(Data == "Counterfactuals" & Class != target_class),] 
  } else {
    df_orig
  }
}

# Execute in parallel
dsets = c('twomoons', 'cassini', 'smiley', 'shapes')
df_arf = foreach(d = dsets, .combine = rbind) %dopar% sim_fun(2000, d, method = "CountARF")
df_moc = foreach(d = dsets, .combine = rbind) %dopar% sim_fun(2000, d, method = "MOC")


# Set scales free but fix x-axis ticks
df = rbind(df_arf, df_moc)
df[Data == "Train", Data := paste0("Train_", Class)]


# Scatter plot
ggplot(df, aes(x = X, y = Y, color = Data)) + 
  geom_point(alpha = 0.75) + 
  scale_color_ordinal() + 
  facet_grid(Method ~ Dataset) + 
  theme_bw() + 
  theme(text = element_text(size = 14))




# ggsave(paste0("examples", ".pdf"), width = 8, height = 4)
