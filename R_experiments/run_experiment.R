library(batchtools)
library(devtools)
library(ggplot2)
library(patchwork)
library(foreach)
load_all("counterfactuals_R_package/")
data.table::setDTthreads(1L)

set.seed(42)

repls = 1L
multicore = TRUE

# Hyperpara
weight_coverage = c(1, 5, 20)
weight_proximity = c(1, 5, 20)
node_selector = c("coverage", "coverage_proximity")
n_synth = c(20L, 200L)
num_x_interest = 10L

# Eval strategies
complex_evaluation = TRUE

# Datasets
datanams = c("pawelczyk", "cassini", "two_sines",
  "bn_20", paste("bn", c(5, 10, 50), "v2", sep = "_"))

# Registry ----------------------------------------------------------------
reg_name = "test"
if (!file.exists("registries")) dir.create("registries")
reg_dir = file.path("registries", reg_name)
unlink(reg_dir, recursive = TRUE)
reg = makeExperimentRegistry(file.dir = reg_dir, seed = 42, 
  packages = c("mlr3verse", "mlr3oml", "iml", "arf", "tictoc",
    "counterfactuals", "xgboost", "data.table", "foreach"), 
  source = c("R_experiments/utils_experiment.R"))
if (multicore) {
	reg$cluster.functions = makeClusterFunctionsMulticore(14L)
}

# Problems -----------------------------------------------------------
get_data = function(data, job, id) {
  # get model & data
  if (id %in% c("bn_20","bn_5_v2", "bn_10_v2", "bn_50_v2")) {
    data_path = "python/synthetic_v2/"
  }  else {
    data_path = "python/synthetic/"
  }
  xgbmodel =  xgboost::xgb.load(file.path(data_path, "xgboost_paras", paste0(id, ".model")))
  x_interests = read.csv(file.path(data_path, "x_interests", paste0(id, ".csv")))
  dt = fread(file.path(data_path, "data/", paste0(id, ".csv")), header = TRUE)
  
  # get desired class
  predictor = Predictor$new(model = xgbmodel, data = dt, y = "y", predict.function = predict.xgboost)
  predictor$task = "classification"
  
  # arf 
  dt = as.data.table(dt)
  dt[, y := NULL]
  dt[, yhat := predictor$predict(dt)]
  mtry = max(2, floor(sqrt(ncol(dt))))
  arf = arf::adversarial_rf(x = dt, parallel = FALSE, mtry = mtry)
  psi = arf::forde(arf, dt, parallel = FALSE)
  
  # Return task, predictor, etc.
  list(dt = dt, predictor = predictor, x_interests = x_interests, arf = arf, psi = psi)
}
addProblem(name = "sim", fun = get_data, seed = 43)

# Algorithms -----------------------------------------------------------
cfs = function(data, job, instance, cf_method, weight_coverage, weight_proximity, n_synth, node_selector, ...) {
  target_class = "pred"
  if (n_synth == "NA") n_synth = NA
  n_synth = as.numeric(n_synth)
  
  p = instance$predictor$data$n.features
  max_feats_to_change = min(ceiling(sqrt(p) + 3), p)
  
  # TODO: loop for x_interests
  iters = min(nrow(instance$x_interests), num_x_interest)
  foreach(idx = seq_len(iters), .combine = rbind) %do% {
    
    x_interest = instance$x_interests[idx,]
    
    # derive target_class & probs
    pred = instance$predictor$predict(x_interest)
    if (pred < 0.5) {
      desired_prob = c(0.5, 1)
    } else {
      desired_prob = c(0, 0.5)
    }
    
    # Generate counterfactuals, coverage only
    tic()
    if (cf_method == "MOCARF") {
      cac = MOCClassif$new(predictor = instance$predictor, plausibility_measure = "lik", max_changed = max_feats_to_change,
        conditional_mutator = "arf_multi", arf = instance$arf, psi = instance$psi, 
        n_generations = 50L, return_all = TRUE, distance_function = "gower_c", quiet = TRUE)
    } else if (cf_method == "MOC") {
      cac = MOCClassif$new(predictor = instance$predictor, n_generations = 50L, 
        max_changed = max_feats_to_change, return_all = TRUE, distance_function = "gower_c", quiet = TRUE)
    } else if (cf_method == "MOCCTREE") {
      cac = MOCClassif$new(predictor = instance$predictor, plausibility_measure = "gower", max_changed = max_feats_to_change,
        conditional_mutator = "ctree", arf = instance$arf, psi = instance$psi, 
        n_generations = 50L, return_all = TRUE, distance_function = "gower_c", quiet = TRUE)
    } else if (cf_method == "ARF") {
      cac = CountARFactualClassif$new(predictor = instance$predictor, max_feats_to_change = max_feats_to_change,
        weight_node_selector = c(weight_coverage, weight_proximity), arf = instance$arf, 
        n_synth = n_synth, node_selector = node_selector, psi = instance$psi)
    } else if (cf_method == "NICE") {
      cac = NICEClassif$new(predictor = instance$predictor, optimization = "plausibility", x_nn_correct = FALSE,
        return_multiple = TRUE, finish_early = FALSE, distance_function = "gower_c")
    } else if (cf_method == "WhatIf") {
      cac = WhatIfClassif$new(predictor = instance$predictor, n_counterfactuals = 50*20L, distance_function = "gower_c")
    }
    cfexpobj = cac$find_counterfactuals(x_interest, 
      desired_class = target_class, 
      desired_prob = desired_prob)
    exectime = toc()

    # Subset to only valid counterfactuals
    cfexpobj$subset_to_valid()
    
    # Either eval based on Gower distance to train data or neg likelihood (ARF-based)
    if (cf_method %in% c("ARF", "MOCARF")) {
      plausibility_measure = "lik"
      nondom_measures = c("dist_x_interest", "no_changed", "neg_lik")
    } else if (cf_method %in% c("MOC", "MOCCTREE")) {
      plausibility_measure = "gower"
      nondom_measures = c("dist_x_interest", "no_changed", "dist_train")
    } else if (cf_method == "NICE") {
      plausibility_measure = "reward"
      nondom_measures = c("dist_x_interest", "no_changed", "reward")
    } else if (cf_method == "WhatIf") {
      plausibility_measure = "gower"
      nondom_measures = c("dist_x_interest", "no_changed")
    }
    
    # Evaluate single counterfactuals based on measures
    eval_measures = c("dist_x_interest", "no_changed", "neg_lik", "dist_train")
    res = cfexpobj$evaluate(arf = instance$arf, measures = eval_measures, 
      psi = instance$psi)
    
    # Aggregate eval of single counterfactuals by averaging
    res_all = res[, lapply(.SD, mean), .SDcols = eval_measures]
    colnames(res_all) = paste0(colnames(res_all), "_all")
    res = cbind(res, res_all)
    
    
    if (complex_evaluation) {
      
      # Aggregate eval nondom only
      res[, nondom := FALSE]
      
      if (cf_method == "NICE") {
        archive = unique(rbindlist(cac$archive))[, pred := NULL]
        res = merge(res, archive, by = instance$predictor$data$feature.names)
        res[, reward := -reward]
      }
      res[miesmuschel:::nondominated(-as.matrix(res[, ..nondom_measures]))$front, nondom := TRUE]
      if (cf_method == "NICE") res = res[, reward := NULL]
      res_nondom = res[nondom == TRUE, lapply(.SD, mean), .SDcols = eval_measures]
      colnames(res_nondom) = paste0(colnames(res_nondom), "_nondom")
      res = cbind(res, res_nondom)
      
      # Evaluate counterfactual set 
      # res_set = cfexpobj$evaluate_set(plausibility_measure = plausibility_measure, 
      #   arf = instance$arf, psi = instance$psi)
      # res = cbind(res, res_set)
    }
    
    # Return all evaluation measures
    res[, id := idx]
    res[, runtime := exectime$toc - exectime$tic]
    res
  }
}
addAlgorithm(name = "cfs", fun = cfs)

# Experiments -----------------------------------------------------------
prob_design = list(
  sim = expand.grid(
    id = datanams
  )
)

algo_design = list(
  cfs = rbind(
    ## ARF with different n_synth
    data.frame(expand.grid(n_synth = n_synth), node_selector = "coverage", 
      weight_coverage = 0, weight_proximity = 0, cf_method = c("ARF")),
    ## MOCARF, ARF sampler + plausibility based on lik
    data.frame(n_synth = "NA", node_selector = "coverage", 
      weight_coverage = 0, weight_proximity = 0, cf_method = c("MOCARF")),
    ## Standard MOC without conditional sampler + plausibility based on gower
    data.frame(n_synth = "NA", node_selector = "NA", 
      weight_coverage = "NA", weight_proximity = "NA",  cf_method = c("MOC")),
    ## MOC with ctree conditional sampler + plausibility based on gower
    data.frame(n_synth = "NA", node_selector = "NA", 
      weight_coverage = "NA", weight_proximity = "NA",  cf_method = c("MOCCTREE")),
   ## NICE with plausibility
  data.frame(n_synth = "NA", node_selector = "NA", 
    weight_coverage = "NA", weight_proximity = "NA",  cf_method = c("NICE")),
  ## WhatIf with plausibility
    data.frame(n_synth = "NA", node_selector = "NA",
      weight_coverage = "NA", weight_proximity = "NA",  cf_method = c("WhatIf"))
  )
)

addExperiments(prob_design, algo_design, repls = repls)
summarizeExperiments()
unwrap(getJobPars())

# Test jobs -----------------------------------------------------------
# testJob(id = 1L)


# Submit -----------------------------------------------------------
submitJobs()
waitForJobs()

# # Get results -------------------------------------------------------------
# res =  flatten(ijoin(reduceResultsDataTable(), getJobPars()))
# 
# res[, method := paste(node_selector, weight_coverage, weight_proximity)]
# res[, dataset := i]
# 
# cols <- c("diversity", "no_nondom", 
#   "frac_nondom", "hypervolume", "dist_x_interest_all", "no_changed_all", 
#   "neg_lik_all", "dist_x_interest_nondom", "no_changed_nondom", 
#   "neg_lik_nondom")
# res_mean <- res[, lapply(.SD, mean), .SDcols = cols, by = .(method, dataset)]
# 
# saveRDS(res, "res.Rds")
# saveRDS(res_mean, "res_mean.Rds")
# 
# # Plot results -------------------------------------------------------------
# res_mean <- readRDS("res_mean.Rds")
# 
# # Likelihood
# p1 = ggplot(res_mean, aes(x = method, y = log(neg_lik_all))) +
#   geom_boxplot() + 
#   theme_bw() + 
#   coord_flip()
# p2 = ggplot(res_mean, aes(x = method, y = log(neg_lik_nondom))) +
#   geom_boxplot() + 
#   theme_bw() + 
#   coord_flip()
# p1 / p2
# 
# # Distance to x_interest
# p1 = ggplot(res_mean, aes(x = method, y = dist_x_interest_all)) +
#   geom_boxplot() + 
#   theme_bw() + 
#   coord_flip()
# p2 = ggplot(res_mean, aes(x = method, y = dist_x_interest_nondom)) +
#   geom_boxplot() + 
#   theme_bw() + 
#   coord_flip()
# p1 / p2
# 
# # Number non-dominated 
# ggplot(res_mean, aes(x = method, y = no_nondom)) +
#   geom_boxplot() + 
#   theme_bw() + 
#   coord_flip()
# 
# # Hypervolume
# ggplot(res_mean, aes(x = method, y = hypervolume)) +
#   geom_boxplot() + 
#   theme_bw() + 
#   coord_flip()
# 
# 
