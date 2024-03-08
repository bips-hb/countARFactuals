devtools::load_all("counterfactuals_R_package/")
library(iml)
library(arf) # using version 0.2.2 PR#20 
library(ggplot2)
library(foreach)

set.seed(1)

# read data
coffee = read.csv("real_world_example/coffee_data_cleaned.csv")
coffee$quality = as.factor(coffee$quality)
coffee$processing.method  = as.factor(coffee$processing.method)
coffee$variety  = as.factor(coffee$variety)
coffee$color  = as.factor(coffee$color)
## uncomment to fix bug! 
# coffee$country.of.origin = as.factor(coffee$country.of.origin)

coffee$cupper.points = NULL
coffee$clean.cup = NULL

id = 577

# sample instance of interest
#x_int = coffee[sample(which(coffee$quality == "bad"), 1),]
x_int = coffee[id,]

# Fit model, exclude x_int
rf = randomForest::randomForest(quality ~ ., coffee[-id,], probability = TRUE)

test = read.csv(file = "error_row.csv", header = TRUE)
test$quality = "good"
test_dt = rbind(coffee, test)

predict(rf, newdata = test_dt[nrow(test_dt), ], type = "prob")[,"good"]
predict(rf, newdata = test_dt, type = "prob")[,"good"][nrow(test_dt)]

