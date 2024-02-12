
library(ranger)
# remotes::install_github("bips-hb/arf") # required for conditional sampling
library(arf)
library(data.table)
library(fastshap)

set.seed(2024)

data(german, package = "rchallenge")  
credit = as.data.table(german[, c("duration", "amount", "purpose", "age", "employment_duration", "housing", "number_credits", "credit_risk")])
credit[, duration := as.numeric(duration)]
credit[, amount := as.numeric(amount)]
credit[, age := as.numeric(age)]

idx <- 998L
x_interest = credit[idx, ]

target_prob <- 0.6
feats_to_change <- 1

# Fit model without obs of interest
rf <- ranger(credit_risk ~ ., credit[-idx, ], probability = TRUE)
predict(rf, x_interest)$prediction

# Shapley values as local importance
pfun <- function(object, newdata) { 
  unname(predict(object, data = newdata)$predictions[, "good"])
}
shap <- explain(rf, X = credit[-idx, -8], pred_wrapper = pfun, newdata = x_interest[, -8],
                nsim = 1000)
vim <- abs(shap[1, ])
ordered_features <- names(sort(vim)) # Smallest (= less important) first

# Fit ARF
dat <- copy(credit)
dat[, credit_risk := NULL]
dat[, yhat := predict(rf, dat)$prediction[, "good"]]
arf <- adversarial_rf(dat, always.split.variables = "yhat")
psi <- forde(arf, dat)

# Conditional sampling
x_interest[, credit_risk := NULL]
x_interest[, yhat := target_prob] 
cols <- c("yhat", ordered_features[1:(length(ordered_features)-feats_to_change)])
fixed <- x_interest[, ..cols]
evidence <- arf:::prep_evi(psi, fixed)
evidence[variable == "yhat", relation := ">="]
synth <- forge(psi, 10, evidence = evidence)
# TODO: Better to condition on yhat >= target_prob (already possible)


# Keep only valid counterfactuals
cfs <- synth[predict(rf, synth)$predictions[, "good"] >= target_prob, ]
#cfs[, yhat := NULL]

# Show obs of interest and the counterfactuals
x_interest
cfs
