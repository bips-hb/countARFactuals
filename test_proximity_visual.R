

library(data.table)
library(arf)
library(fdm2id)
library(mlbench)
library(ggplot2)
library(ranger)
library(ggsci)
#library(doParallel)
#registerDoParallel(2)

set.seed(2024)

n_trn <- 2000
dataset <- "twomoons"
target_prob <- 0.8
num_cfs <- 500

# Simulate data
if (dataset == 'twomoons') {
  x = data.twomoons(n = n_trn/2, graph = FALSE, seed = runif(1, 1, 10000))
  x$Class = as.factor(gsub('Class ', '', x$Class))
} else {
  if (dataset == 'cassini') {
    tmp = mlbench.cassini(n_trn)
  } else if (dataset == 'smiley') {
    tmp = mlbench.smiley(n_trn)
  } else if (dataset == 'shapes') {
    tmp = mlbench.shapes(n_trn)
  }
  x = data.frame(tmp$x, as.factor(tmp$classes))
  colnames(x) = c('X', 'Y', 'Class')
}
x <- as.data.table(x)

# Select a point of interest
x_interest <- data.table(X = 1.5, Y = 0, yhat = target_prob)
#x_interest <- data.table(X = -.5, Y = -.2, yhat = target_prob)

# Fit model without obs of interest
rf = ranger(Class ~ ., x, probability = TRUE)

# Fit ARF
dat <- copy(x)
dat[, Class := NULL]
dat[, yhat := predict(rf, dat)$prediction[, "2"]]
arf <- adversarial_rf(dat, always.split.variables = "yhat", parallel = FALSE)
psi <- forde(arf, dat, parallel = FALSE)

# Gower distances
leaf_means <- dcast(psi$cnt[variable != "yhat", .(f_idx, variable, mu)], f_idx ~ variable, value.var = "mu")
leaf_dist <- data.table(f_idx = leaf_means$f_idx, dist = gower:::gower_dist(leaf_means, x_interest))

# Use weighted combination of coverage and weights as new leaf weights
psi$forest <- merge(psi$forest, leaf_dist, by = "f_idx")
psi$forest[, cvg := exp(20*cvg-20*dist)]
psi$forest[, dist := NULL]

# Change only x
cols <- c("yhat", "Y")
fixed <- x_interest[, ..cols]
evidence <- arf:::prep_evi(psi, fixed)
evidence[variable == "yhat", relation := ">="]
synth <- forge(psi, num_cfs, evidence = evidence)
cfs_x <- synth[predict(rf, synth)$predictions[, "2"] >= target_prob, ]

# Change only y
cols <- c("yhat", "X")
fixed <- x_interest[, ..cols]
evidence <- arf:::prep_evi(psi, fixed)
evidence[variable == "yhat", relation := ">="]
synth <- forge(psi, num_cfs, evidence = evidence)
cfs_y <- synth[predict(rf, synth)$predictions[, "2"] >= target_prob, ]

# Change x and y
cols <- c("yhat")
fixed <- x_interest[, ..cols]
evidence <- arf:::prep_evi(psi, fixed)
evidence[variable == "yhat", relation := ">="]
synth <- forge(psi, num_cfs, evidence = evidence)
cfs_xy <- synth[predict(rf, synth)$predictions[, "2"] >= target_prob, ]

# Put all together
cfs_x[, Class := as.factor("CF_x")]
cfs_y[, Class := as.factor("CF_y")]
cfs_xy[, Class := as.factor("CF_xy")]
x_interest[, Class := as.factor("X*")]
df <- rbind(rbind(rbind(rbind(x, 
                              x_interest[, .(X, Y, Class)]), 
                        cfs_x[, .(X, Y, Class)]), 
                  cfs_y[, .(X, Y, Class)]), 
            cfs_xy[, .(X, Y, Class)])

# Scatter plot
ggplot(df, aes(x = X, y = Y, color = Class)) + 
  geom_point(alpha = 0.75) + 
  scale_color_ordinal() + 
  theme_bw()

