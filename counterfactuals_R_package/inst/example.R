library("randomForest")



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
library("arf")
dat[, Species := NULL]
dat[, yhat := predictor$predict(dat)[,"virginica"]]

frst = adversarial_rf(dat, always.split.variables = "yhat")
fd = forde(frst, dat)

# Find counterfactuals for x_interest
load_all()

moc_classif = MOCClassif$new(predictor, n_generations = 15L, 
  quiet = TRUE, conditional_mutator = "arf_multi", plausibility_measure = "lik", 
  arf = frst)

set.seed(1234L)
set.seed(1234567L)
cfactuals = moc_classif$find_counterfactuals(
  x_interest = x_interest, desired_class = "virginica", desired_prob = c(0, 0.5)
)

cfactuals$evaluate_set(plausbility_measure = "lik", arf = frst)

# Print the counterfactuals
cfactuals$data
# Plot evolution of hypervolume and mean and minimum objective values
moc_classif$plot_statistics()



# ARF 
arf_classif = CountARFactualClassif$new(predictor, arf = frst)

cfactuals = arf_classif$find_counterfactuals(
  x_interest = iris[150L, ], desired_class = "virginica", desired_prob = c(0, .5)
)
# Print the counterfactuals
cfactuals$data
# Plot evolution of hypervolume and mean and minimum objective values
cfactuals$evaluate_set(plausbility_measure = "lik", arf = frst)
cfactuals$plot_parallel()


