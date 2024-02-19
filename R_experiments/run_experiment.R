
library(batchtools)
library(ggplot2)
library(patchwork)
load_all("counterfactuals_R_package/")

set.seed(42)

repls = 1L
weight_coverage = c(1, 5, 20)
weight_proximity = c(1, 5, 20)
node_selector = c("coverage", "coverage_proximity")
n_synth = c(10L, 200L)
num_x_interest = 10L
datanams = c("pawelczyk", "cassini", "two_sines" ,"bn_1")

# Registry ----------------------------------------------------------------
reg_name = "evaluate_simulation"
if (!file.exists("registries")) dir.create("registries")
reg_dir = file.path("registries", reg_name)
unlink(reg_dir, recursive = TRUE)
makeExperimentRegistry(file.dir = reg_dir, seed = 42, 
  packages = c("mlr3verse", "mlr3oml", "iml", "arf", 
    "counterfactuals", "xgboost"), source = c("R_experiments/utils_experiment.R"))

# Problems -----------------------------------------------------------
get_data = function(data, job, id) {
  # Get data
  # get model & data
  xgbmodel =  xgboost::xgb.load(file.path("python/synthetic/xgboost_paras", paste0(id, ".model")))
  x_interests = read.csv(file.path("python/synthetic/x_interests", paste0(id, ".csv")))
  dt = fread(file.path("python/synthetic/data/", paste0(id, ".csv")), header = TRUE)
  
  # get desired class
  predictor = Predictor$new(model = xgbmodel, data = dt, y = "y", predict.function = predict.xgboost)
  predictor$task = "classification"
  
  # arf 
  dt = as.data.table(dt)
  dt[, y := NULL]
  dt[, yhat := predictor$predict(dt)]
  arf = arf::adversarial_rf(x = dt, parallel = FALSE)
  
  # Return task, predictor, etc.
  list(dt = dt, predictor = predictor, x_interests = x_interests, arf = arf)
}
addProblem(name = "sim", fun = get_data, seed = 43)

# Algorithms -----------------------------------------------------------
cfs = function(data, job, instance, cf_method, weight_coverage, weight_proximity, n_synth, node_selector, ...) {
  
  target_class = "pred"
  n_synth = as.numeric(n_synth)
  
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
    if (cf_method == "MOCARF") {
      cac = MOCClassif$new(predictor = instance$predictor, plausibility_measure = "lik", 
        conditional_mutator = "arf_multi", arf = instance$arf)
    } else if (cf_method == "MOC") {
      cac = MOCClassif$new(predictor = instance$predictor)
    } else if (cf_method == "ARF") {
      cac = CountARFactualClassif$new(predictor = instance$predictor, 
        weight_node_selector = c(weight_coverage, weight_proximity), arf = instance$arf, 
        n_synth = n_synth, node_selector = node_selector)
    }
    cfexpobj = cac$find_counterfactuals(x_interest, 
      desired_class = target_class, 
      desired_prob = desired_prob)
    cfexpobj$subset_to_valid()
    
    # Evaluate counterfactuals, use ARF for plausibility evaluation
    res_set = cfexpobj$evaluate_set(plausibility_measure = "lik", arf = instance$arf)
    
    # Find non-dominated CFs
    measures = c("dist_x_interest", "no_changed", "neg_lik")
    cfexp = cfexpobj$evaluate(arf = instance$arf, measures = c(measures, "dist_train"))
    cfexp[, nondom := miesmuschel::rank_nondominated(-as.matrix(cfexp[, ..measures]))$fronts == 1]
    
    # Evaluate all and non-dominated only
    res_all = cfexp[, lapply(.SD, mean), .SDcols = measures]
    res_nondom = cfexp[nondom == TRUE, lapply(.SD, mean), .SDcols = measures]
    colnames(res_all) = paste0(colnames(res_all), "_all")
    colnames(res_nondom) = paste0(colnames(res_nondom), "_nondom")
    
    # Return all evaluation measures
    cbind(cfexp, res_set, res_all, res_nondom)[, id := idx]
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
    ## Standard ARF without conditional sampler + plausibility based on gower
    data.frame(n_synth = "NA", node_selector = "NA", 
      weight_coverage = "NA", weight_proximity = "NA",  cf_method = c("MOC")))
)

addExperiments(prob_design, algo_design, repls = repls)
summarizeExperiments()
unwrap(getJobPars())

# Test jobs -----------------------------------------------------------
testJob(id = 1)

# Submit -----------------------------------------------------------
submitJobs()
waitForJobs()

# Get results -------------------------------------------------------------
res =  flatten(ijoin(reduceResultsDataTable(), getJobPars()))

res[, method := paste(node_selector, weight_coverage, weight_proximity)]
res[, dataset := i]

cols <- c("diversity", "no_nondom", 
  "frac_nondom", "hypervolume", "dist_x_interest_all", "no_changed_all", 
  "neg_lik_all", "dist_x_interest_nondom", "no_changed_nondom", 
  "neg_lik_nondom")
res_mean <- res[, lapply(.SD, mean), .SDcols = cols, by = .(method, dataset)]

saveRDS(res, "res.Rds")
saveRDS(res_mean, "res_mean.Rds")

# Plot results -------------------------------------------------------------
res_mean <- readRDS("res_mean.Rds")

# Likelihood
p1 = ggplot(res_mean, aes(x = method, y = log(neg_lik_all))) +
  geom_boxplot() + 
  theme_bw() + 
  coord_flip()
p2 = ggplot(res_mean, aes(x = method, y = log(neg_lik_nondom))) +
  geom_boxplot() + 
  theme_bw() + 
  coord_flip()
p1 / p2

# Distance to x_interest
p1 = ggplot(res_mean, aes(x = method, y = dist_x_interest_all)) +
  geom_boxplot() + 
  theme_bw() + 
  coord_flip()
p2 = ggplot(res_mean, aes(x = method, y = dist_x_interest_nondom)) +
  geom_boxplot() + 
  theme_bw() + 
  coord_flip()
p1 / p2

# Number non-dominated 
ggplot(res_mean, aes(x = method, y = no_nondom)) +
  geom_boxplot() + 
  theme_bw() + 
  coord_flip()

# Hypervolume
ggplot(res_mean, aes(x = method, y = hypervolume)) +
  geom_boxplot() + 
  theme_bw() + 
  coord_flip()


