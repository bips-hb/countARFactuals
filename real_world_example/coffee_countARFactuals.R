#library(counterfactuals)
devtools::load_all("counterfactuals_R_package/")
library(iml)
library(arf) # requires CRAN version 0.2.0 
library(ggplot2)

set.seed(2024)

# read data
coffee = read.csv("real_world_example/coffee_data_cleaned.csv")
coffee$quality = as.factor(coffee$quality)

# define actionable features
actionable_features = c("country.of.origin",
                        "harvest.year",
                        "variety",
                        "processing.method",
                        "moisture",
                        "color", 
                        "acidity",
                        "sweetness",
                        "altitude_mean_meters",
                        "quality")

# sample instance of interest
x_int = coffee[sample(which(coffee$quality == "bad"), 1),]
     
# Fit model, exclude x_int
rf = randomForest::randomForest(quality ~ ., coffee[-as.numeric(rownames(x_int)),], probability = TRUE)

# generate countARFactuals
predictor = Predictor$new(rf, type = "prob" )
countARFactual_classif = CountARFactualClassif$new(predictor, 
                                                  # fixed_features =  names(coffee)[!names(coffee) %in% actionable_features], 
                                                   max_feats_to_change = 3) # change max 3 features , max_feats_to_change = 3
# Hyperparameters

# weight_coverage = c(1, 5, 20)
# weight_proximity = c(1, 5, 20)
# node_selector = c("coverage", "coverage_proximity")
# n_synth = c(20L, 200L)
# max_feats_to_change = 3

weight_coverage = c(5)
weight_proximity = c(20)
node_selector = c("coverage_proximity")
n_synth = c(200L)
max_feats_to_change = 3

cac = CountARFactualClassif$new(predictor = predictor, max_feats_to_change = max_feats_to_change,
                                weight_node_selector = c(weight_coverage, weight_proximity), 
                                n_synth = n_synth, node_selector = node_selector)

my_countARFactuals = cac$find_counterfactuals(x_interest = x_int, desired_class = "good", desired_prob = c(0.5,1)) 

#my_countARFactuals = countARFactual_classif$find_counterfactuals(x_interest = x_int, desired_class = "good", desired_prob = c(0.5,1)) 

# have a look
my_countARFactuals$data

# taking only valid countARFactuals into account
my_countARFactuals$subset_to_valid()

 
# taking only nondominated countARFactuals into account
# plausibility measure: dist_train
objectives = c("dist_target", "dist_x_interest", "no_changed", "dist_train") 
res = my_countARFactuals$evaluate(objectives)[, objectives, with = FALSE]
idnondom = miesmuschel::rank_nondominated(-as.matrix(res))$fronts == 1
res_nondom = my_countARFactuals$data[idnondom, ]



# # visualize results:
# # plot frequency
# my_countARFactuals$plot_freq_of_feature_changes(subset_zero = TRUE,
#       feature_names = names(my_countARFactuals$get_freq_of_feature_changes(subset_zero = TRUE)[1:5]))

# parallel plot of top 5 features
my_countARFactuals$plot_parallel(feature_names = names(
  my_countARFactuals$get_freq_of_feature_changes()[1:5]),  digits_min_max = 2L,
  )
# ggsave("plt_parallel.png", plot = last_plot(), device = "png",path = "./real_world_example")

# plot surface
my_countARFactuals$plot_surface(feature_names = c('acidity', 'processing.method'))
# ggsave("plt_surface.png", plot = last_plot(), device = "png",path = "./real_world_example")


res_nondom
x_int


