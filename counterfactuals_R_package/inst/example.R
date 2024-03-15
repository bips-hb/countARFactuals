library("randomForest")
library("data.table")
library("devtools")
library("arf")
library("foreach")
load_all("counterfactuals_R_package/")

set.seed(1234L)

data(iris, package = "datasets")
x_interest = iris[150L, ]
dat = data.table(iris[-150L, ])

# Train a model
rf = randomForest(Species ~ ., data = dat)
# Create a predictor object
predictor = iml::Predictor$new(rf, type = "prob")
predictor$predict(x_interest)

dat = as.data.table(dat)

# Fit an ARF

dat[, Species := NULL]
dat[, yhat := predictor$predict(dat)[,"virginica"]]

frst = adversarial_rf(dat, always.split.variables = "yhat")
fd = forde(frst, dat)

# Find counterfactuals for x_interest
moc_classif = MOCClassif$new(predictor, n_generations = 5L, 
  quiet = TRUE, conditional_mutator = "ctree", plausibility_measure = "gower", 
  arf = frst, return_all = FALSE)

set.seed(123456L)
cfactuals = moc_classif$find_counterfactuals(
  x_interest = x_interest, desired_class = "virginica", desired_prob = c(0, 0.5)
)
cfactuals
cfactuals$evaluate_set(plausibility_measure = "lik", arf = frst)
# Print the counterfactuals
cfactuals$data
# Plot evolution of hypervolume and mean and minimum objective values
moc_classif$plot_statistics()


# ARF 
arf_classif = CountARFactualClassif$new(predictor, arf = frst,weight_node_selector = c(20, 20))
arf_classif = CountARFactualClassif$new(predictor, 
  arf = frst,weight_node_selector = c(20, 20), 
  fixed_features = NULL, 
  max_feats_to_change = 4L)
cfactuals = arf_classif$find_counterfactuals(
  x_interest = iris[150L, ], desired_class = "virginica", desired_prob = c(0, .5)
)
cfactuals$evaluate(measures = "no_changed")
arf_classif = CountARFactualClassif$new(predictor, arf = frst, node_selector = "coverage")
cfactuals$subset_to_valid()
cfactuals$evaluate_set(plausibility_measure = "lik", arf = frst)
colMeans(cfactuals$evaluate(arf = frst))
cfactuals$plot_surface(feature_names = c("Petal.Width", "Petal.Length"))
cfactuals$plot_parallel()

