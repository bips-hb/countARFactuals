#library(counterfactuals)
devtools::load_all("counterfactuals_R_package/")
library(iml)
library(arf) # requires CRAN version 0.2.0 
set.seed(1)

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
my_countARFactuals = countARFactual_classif$find_counterfactuals(x_interest = x_int, desired_class = "good", desired_prob = c(0.5,1)) 

# have a look
my_countARFactuals$data

  
