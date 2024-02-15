
library(batchtools)
library(ggplot2)
library(patchwork)

set.seed(42)

task_ids = 1:6 #1:13
weight_coverage = c(0, 1, 5, 20)
weight_proximity = c(0, 1, 5, 20)

# Registry ----------------------------------------------------------------
reg_name = "evaluate_cc18"
if (!file.exists("registries")) dir.create("registries")
reg_dir = file.path("registries", reg_name)
unlink(reg_dir, recursive = TRUE)
makeExperimentRegistry(file.dir = reg_dir, seed = 42, 
                       packages = c("mlr3verse", "mlr3oml", "iml", "arf", 
                                    "counterfactuals"))

# Problems -----------------------------------------------------------
get_data = function(data, job, id) {
  # Get data
  tasks_binary = readRDS("tasks_binary.Rds")
  task = tasks_binary[[id]]
  
  # Select random point of interest
  idx = sample(1:task$nrow, size = 1L)
  x_interest = cbind(task$data(rows = idx, cols = task$feature_names), 
                      task$data(rows = idx, cols = task$target_names))
  task$filter(setdiff(1:task$nrow, idx))
  
  # Fit model
  learner = lrn("classif.ranger", predict_type = "prob")
  learner$train(task)
  predictor = Predictor$new(model = learner, data = task$data(), y = task$target_names)
  
  # Get desired class
  pred = predictor$predict(x_interest)
  target_class = names(which.min(pred))
  
  # Return task, predictor, etc.
  list(task = task, predictor = predictor, x_interest = x_interest, 
       target_class = target_class)
}
addProblem(name = "cc18_binary", fun = get_data, seed = 43)

# Algorithms -----------------------------------------------------------
arf_cfs = function(data, job, instance, weight_coverage, weight_proximity, ...) {
  # Generate counterfactuals, coverage only
  cac = CountARFactualClassif$new(predictor = instance$predictor, 
                                  weight_node_selector = c(weight_coverage, weight_proximity), 
                                  ...)
  cfexpobj = cac$find_counterfactuals(instance$x_interest, 
                                      desired_class = instance$target_class, 
                                      desired_prob = c(0.5, 1))
  cfexpobj$subset_to_valid()
  
  # Evaluate counterfactuals, use ARF for plausibility evaluation
  arf = adversarial_rf(instance$task$data(cols = instance$task$feature_names))
  res_set = as.matrix(cfexpobj$evaluate_set(plausibility_measure = "lik", arf = arf))[1, ]
  
  # Find non-dominated CFs
  measures = c("dist_x_interest", "no_changed", "neg_lik")
  cfexp = cfexpobj$evaluate(arf = arf, measures = c(measures, "dist_train"))
  cfexp[, nondom := miesmuschel::rank_nondominated(-as.matrix(cfexp[, ..measures]))$fronts == 1]
  
  # Evaluate all and non-dominated only
  res_all = as.matrix(cfexp[, lapply(.SD, mean), .SDcols = measures])[1, ]
  res_nondom = as.matrix(cfexp[nondom == TRUE, lapply(.SD, mean), .SDcols = measures])[1, ]
  names(res_all) = paste0(names(res_all), "_all")
  names(res_nondom) = paste0(names(res_nondom), "_nondom")
  
  # Return all evaluation measures
  c(res_set, res_all, res_nondom)
}
addAlgorithm(name = "CountARF", fun = arf_cfs)

# Experiments -----------------------------------------------------------
prob_design = list(
  cc18_binary = expand.grid(
    i = task_ids
  )
)

algo_design = list(
  CountARF = rbind(cbind(expand.grid(weight_coverage = weight_coverage, 
                         weight_proximity = weight_proximity), 
                         node_selector = "coverage_proximity"), 
                   data.frame(weight_coverage = 0, weight_proximity = 0, 
                              node_selector = "coverage"))
)

addExperiments(prob_design, algo_design, repls = 5)
summarizeExperiments()

# Test jobs -----------------------------------------------------------
testJob(id = 1)

# Submit -----------------------------------------------------------
submitJobs()
waitForJobs()

# Get results -------------------------------------------------------------
res =  flatten(ijoin(reduceResultsDataTable(), getJobPars()))
#res

# Plot results -------------------------------------------------------------
res[, method := paste(node_selector, weight_coverage, weight_proximity)]
res[, dataset := i]

cols <- c("diversity", "no_nondom", 
          "frac_nondom", "hypervolume", "dist_x_interest_all", "no_changed_all", 
          "neg_lik_all", "dist_x_interest_nondom", "no_changed_nondom", 
          "neg_lik_nondom")
res_mean <- res[, lapply(.SD, mean), .SDcols = cols, by = .(method, dataset)]

# Likelihood
p1 = ggplot(res_mean, aes(x = method, y = neg_lik_all)) +
  geom_boxplot() + 
  theme_bw() + 
  coord_flip()
p2 = ggplot(res_mean, aes(x = method, y = neg_lik_nondom)) +
  geom_boxplot() + 
  theme_bw() + 
  coord_flip()
p1 / p2


