#--- AGGREGATE RESULTS ---
library(data.table)
library(batchtools)


eval_columns = c("dist_x_interest", "no_changed", "neg_lik", "dist_train", 
  "dist_x_interest_nondom", "no_changed_nondom", "neg_lik_nondom", 
  "dist_train_nondom", "log_probs", "runtime")
save_columns = c(eval_columns, "id", "job.id", "n_synth", "problem", "cf_method", "dataset", "nondom")
csv_dir = "cfs/08_03_2"

# Get results -------------------------------------------------------------

res = NULL
files = list.files(csv_dir)
files = files[grepl(files, pattern = "log_probs.csv")]
for (file in files) {
  print(file)
  tmp = fread(file = file.path(csv_dir, file), header = TRUE)[, ..save_columns]
  res = rbind(res, tmp)
}

# Preprocessing ------------------------------------------------------------
setnames(res, "cf_method", "method")
max_features = list(cassini = 2L, pawelczyk = 2L, two_sines = 2L, bn_5_v2 = 4L, bn_10_v2 = 9L, 
  bn_20 = 19L, bn_50_v2 = 49L)
res = res[, max_features := as.numeric(max_features[dataset])]
res = res[, rel_no_changed := no_changed/max_features]
saveRDS(res, "R_experiments/res.Rds")

# # Correlation estimated vs. true implausibility ------------------------------
res = res[, probs := exp(log_probs)]
dt_lik = res[, .(cor_lik = cor(probs, -neg_lik, method = "spearman")), by = .(method, dataset, id)]
dt_gow = res[, .(cor_gow = cor(probs, -dist_train, method = "spearman")), by = .(method, dataset, id)]
res_agg = merge(dt_lik, dt_gow, by = c("method", "dataset", "id"))

# HV on o_prox, o_sparse and true implausibility ------------------------------
res = res[, implausibility := 1-probs]
### nadir per dataset/id 
hv_measures = c("dist_x_interest", "rel_no_changed", "implausibility")

res[, c("scaled_dist_x_interest", "scaled_rel_no_changed", "scaled_implausibility") :=
    lapply(.SD, function(x) as.vector(scale(x))), .SDcols = hv_measures,
  by = .(dataset)]
hv_measures = paste0("scaled_", hv_measures)
nadir = res[, lapply(.SD, max), .SDcols = hv_measures, by = .(dataset)]
setnames(nadir, old = hv_measures,
  new = c("nadir_dist_x_interest", "nadir_no_changed", "nadir_implaus"))
res = merge(res, nadir, by = c("dataset"))

get_hv = function(x) {
  miesmuschel::domhv(fitnesses = -as.matrix(rbind(x[, ..hv_measures])),
    nadir = -as.numeric(x[1, c("nadir_dist_x_interest", "nadir_no_changed", "nadir_implaus")]))
}
hv_columns = c(hv_measures, "nadir_dist_x_interest", "nadir_no_changed", "nadir_implaus")

hvs = res[nondom == TRUE, get_hv(.SD),
  .SDcols = hv_columns, by = .(method, dataset, id)]
setnames(hvs, "V1", "hv_normalized")
res_agg = merge(res_agg, hvs, by = c("method", "dataset", "id"))

# Number of CFEs -------------------------------------------------------------
number = res[nondom == TRUE, .N, by = .(method, dataset, id)]
setnames(number, "N", "number")
res_agg = merge(res_agg, number, by = c("method", "dataset", "id"))

# Runtime --------------------------------------------------------------------
runtime = res[, mean(runtime), by = .(method, dataset, id)]
setnames(runtime, "V1", "runtime")
res_agg = merge(res_agg, runtime)

saveRDS(res_agg, "R_experiments/res_agg.Rds")
