#library(counterfactuals)
devtools::load_all("counterfactuals_R_package/")
library(iml)
library(arf) # requires CRAN version 0.2.0 
library(ggplot2)

set.seed(10)

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

# taking only valid countARFactuals into account
my_countARFactuals$subset_to_valid()

# visualize results:
# plot frequency 
my_countARFactuals$plot_freq_of_feature_changes(subset_zero = TRUE, 
      feature_names = names(my_countARFactuals$get_freq_of_feature_changes(subset_zero = TRUE)[1:5]))

# parallel plot of top 5 features
my_countARFactuals$plot_parallel(feature_names = names(
  my_countARFactuals$get_freq_of_feature_changes()[1:5]),  digits_min_max = 2L,
  )
ggsave("plt_parallel.png", plot = last_plot(), device = "png",path = "./real_world_example")

# plot surface
my_countARFactuals$plot_surface(feature_names = c('aroma', 'acidity'))
ggsave("plt_surface.png", plot = last_plot(), device = "png",path = "./real_world_example")
