devtools::load_all("counterfactuals_R_package/")
library(iml)
library(OpenML)
library(arf) # requires CRAN version 0.2.0 


set.seed(10)
# German Credit dataset
df = getOMLTask(task.id = 31)$input$data.set$data

# sample instance of interest
x_int = df[sample(which(df$class == "bad"), 1),]

# Fit model, exclude x_int
rf = randomForest::randomForest(class ~ ., df[-as.numeric(rownames(x_int)),], probability = TRUE)

# generate countARFactuals
predictor = Predictor$new(rf, type = "prob" )
countARFactual_classif = CountARFactualClassif$new(predictor) # change max 3 features , max_feats_to_change = 3
my_countARFactuals = countARFactual_classif$find_counterfactuals(x_interest = x_int, desired_class = "good", desired_prob = c(0.5,1)) 
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

res_nondom

