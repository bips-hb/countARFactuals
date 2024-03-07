#library(counterfactuals)
devtools::load_all("counterfactuals_R_package/")
library(iml)
library(arf) # using version 0.2.2 PR#20 
library(ggplot2)

set.seed(1)

# read data
coffee = read.csv("real_world_example/coffee_data_cleaned.csv")
coffee$quality = as.factor(coffee$quality)
coffee$processing.method  = as.factor(coffee$processing.method)
coffee$variety  = as.factor(coffee$variety)
coffee$color  = as.factor(coffee$color)

coffee$cupper.points = NULL
coffee$clean.cup = NULL

# define actionable features
actionable_features = c("country.of.origin",
                        "harvest.year",
                        "variety",
                        "processing.method",
                        "moisture",
                   #     "color", 
                   #     "acidity",
                    #    "sweetness",
                        "altitude_mean_meters",
                        "quality")

# sample instance of interest
#x_int = coffee[sample(which(coffee$quality == "bad"), 1),]
x_int = coffee[577,]

# Fit model, exclude x_int
rf = randomForest::randomForest(quality ~ ., coffee[-as.numeric(rownames(x_int)),], probability = TRUE)

# generate countARFactuals
predictor = Predictor$new(rf, type = "prob" )

# Hyperparameters

# Hyperparameters in R experiments
# weight_coverage = c(1, 5, 20)
# weight_proximity = c(1, 5, 20)
# node_selector = c("coverage", "coverage_proximity")
# n_synth = c(20L, 200L)
# max_feats_to_change = 3

weight_coverage = c(1)
weight_proximity = c(5)
node_selector = c("coverage_proximity")
n_synth = c(20L)
max_feats_to_change = 10

cac = CountARFactualClassif$new(predictor = predictor, max_feats_to_change = max_feats_to_change,
                                weight_node_selector = c(weight_coverage, weight_proximity),
                                fixed_features =  names(coffee)[!names(coffee) %in% actionable_features],
                                n_synth = n_synth, node_selector = node_selector)

my_countARFactuals = cac$find_counterfactuals(x_interest = x_int, desired_class = "good", desired_prob = c(0.5,1)) 

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


# visualize results:

# parallel plot of top 5 features
my_countARFactuals$plot_parallel(feature_names = names(
  my_countARFactuals$get_freq_of_feature_changes()[1:5]),  digits_min_max = 2L,
  )
# ggsave("plt_parallel.png", plot = last_plot(), device = "png",path = "./real_world_example")

# plot surface
my_countARFactuals$plot_surface(feature_names = c('variety', 'processing.method'))
# ggsave("plt_surface.png", plot = last_plot(), device = "png",path = "./real_world_example")


res_nondom$quality = predict(rf, res_nondom) # why is this not always "good" quality?! this data is after $subset_to_valid()
x_int = as.data.frame(x_int)
row.names(x_int) = "x_interest"
# write.csv(as.data.frame(rbind(x_int, res_nondom)), "./real_world_example/coffee_countARFactuals.csv")

# Have a look
res_nondom
x_int

