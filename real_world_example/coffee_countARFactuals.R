#library(counterfactuals)
devtools::load_all("counterfactuals_R_package/")
library(iml)

set.seed(2024)

# read data
coffee = read.csv("real_world_example/coffee_data_cleaned.csv")
coffee$quality = as.factor(coffee$quality)
# sample instance of interest
x_int = coffee[sample(which(coffee$quality == "bad"), 1),]
     
# Fit model, exclude x_int
rf = randomForest::randomForest(quality ~ ., coffee[-as.numeric(rownames(x_int)),], probability = TRUE)

# generate countARFactuals
predictor = Predictor$new(rf, type = "prob" )
countARFactual_classif = CountARFactualClassif$new(predictor)

countARFactual_classif$find_counterfactuals(x_interest = x_int, desired_class = "good", desired_prob = c(0.5,1)) # doesn't work
# Error message: Error in rbindlist(l, use.names, fill, idcol) : 
# Item 2 has 3 columns, inconsistent with item 1 which has 4 columns. To fill missing columns use fill=TRUE.



# # for debugging -- check whether it works with Iris example
# rf = randomForest::randomForest(Species ~ ., data = iris)
# predictor = iml::Predictor$new(rf, type = "prob")
# predictor$task = "classification"
# arf_classif = CountARFactualClassif$new(predictor)
# cfactuals = arf_classif$find_counterfactuals(
#   x_interest = iris[150L, ], desired_class = "versicolor", desired_prob = c(0.5, 1)
# ) # doesn't work either
# cfactuals = arf_classif$find_counterfactuals(
#   x_interest = iris[150, ], desired_class = "versicolor", desired_prob = c(0.5, 1)
# ) # doesn't work either -- same error as with
# # for debugging -- check whether MOC works 
# moc_classif = MOCClassif$new(predictor, n_generations = 15L, quiet = TRUE)
# cfactuals = moc_classif$find_counterfactuals(
#   x_interest = iris[150L, ], desired_class = "versicolor", desired_prob = c(0.5, 1))
# # does work
  