# --- VISUALIZE RESULTS ----
library(data.table)
library(batchtools)


eval_columns = c("dist_x_interest", "no_changed", "neg_lik", "dist_train", 
  "dist_x_interest_nondom", "no_changed_nondom", "neg_lik_nondom", 
  "dist_train_nondom", "log_probs", "runtime")
save_columns = c(eval_columns, "id", "job.id", "n_synth", "problem", "cf_method", "dataset", "nondom")
csv_dir = "cfs/29_02"

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
res = res[, log_probs := exp(log_probs)]
res[, method := ifelse(cf_method == "ARF", paste(cf_method, n_synth), cf_method)]
res = res[, implausibility := 1-log_probs]
res = res[, max_no_changed := max(no_changed), by = .(method, dataset, id)]
res = res[, rel_no_changed := no_changed/max(no_changed), by = .(method, dataset, id)]
# saveRDS(res, "R_experiments/res.Rds")

# Averages per objective ---------------------------------------------------
res_mean = res[, lapply(.SD, mean), .SDcols = eval_columns, by = .(method, dataset, id)]
res_log_probs_nondom = res[nondom == TRUE, lapply(.SD, mean), .SDcols = "log_probs", by = .(method, dataset, id)]
setnames(res_log_probs_nondom, "log_probs", "log_probs_nondom")
res_mean = merge(res_mean, res_log_probs_nondom, by = c("method", "dataset", "id"))

# Correlation estimated vs. true implausibility ------------------------------
dt_lik = res[, .(cor_lik = cor(log_probs, -neg_lik, method = "spearman")), by = .(method, dataset, id)]
dt_gow = res[, .(cor_gow = cor(log_probs, -dist_train, method = "spearman")), by = .(method, dataset, id)]
corr = merge(dt_lik, dt_gow, by = c("method", "dataset", "id"))
res_mean = merge(res_mean, corr, by = c("method", "dataset", "id"))

# HV on o_prox, o_sparse and true implausibility ------------------------------
### nadir per dataset/id 
hv_measures = c("dist_x_interest", "rel_no_changed", "implausibility")
hv_columns = c(hv_measures, "nadir_dist_x_interest", "nadir_no_changed", "nadir_implaus")
nadir = res[, lapply(.SD, max), .SDcols = hv_measures, by = .(dataset, id)]
setnames(nadir, old = c("dist_x_interest", "rel_no_changed", "implausibility"),
  new = c("nadir_dist_x_interest", "nadir_no_changed", "nadir_implaus"))
res = merge(res, nadir, by = c("dataset", "id"))

get_hv = function(x) {
  miesmuschel::domhv(fitnesses = -as.matrix(rbind(x[, c("dist_x_interest", "rel_no_changed", "implausibility")])),
    nadir = -as.numeric(x[1, c("nadir_dist_x_interest", "nadir_no_changed", "nadir_implaus")]))
}
hvs = res[, hv := get_hv(.SD), .SDcols = hv_columns, by = .(method, dataset, id)]
hvs_nondom = res[nondom == TRUE, hv_nondom := get_hv(.SD),
  .SDcols = hv_columns, by = .(method, dataset, id)]
hvs_mean = res[, lapply(.SD, mean, na.rm = TRUE), .SDcols = c("hv", "hv_nondom"), by = .(method, dataset, id)]
res_mean = merge(res_mean, hvs_mean, by = c("method", "dataset", "id"))

# Number of CFEs -------------------------------------------------------------
res[, number := .N, by = .(method, dataset, id)]
res[nondom == TRUE, number_nondom := .N, by = .(method, dataset, id)]
no_mean = res[, lapply(.SD, mean, na.rm = TRUE), .SDcols = c("number", "number_nondom"), 
  by = .(method, dataset, id)]
res_mean = merge(res_mean, no_mean, by = c("method", "dataset", "id"))

saveRDS(res_mean, "R_experiments/res_mean_new.Rds")
