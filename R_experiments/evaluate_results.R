# --- VISUALIZE RESULTS ----
library(data.table)
library(batchtools)
library(ggplot2)
library(reshape2)


eval_columns = c("dist_x_interest", "no_changed", "neg_lik", "dist_train", 
  "dist_x_interest_nondom", "no_changed_nondom", "neg_lik_nondom", 
  "dist_train_nondom", "log_probs", "runtime")

csv_dir = "cfs/23_02"

# Get results -------------------------------------------------------------

res = NULL



files = list.files(csv_dir)
files = files[grepl(files, pattern = "log_probs")]
for (file in files) {
  res = rbind(res, fread(file = file.path(csv_dir, file), header = TRUE))
}


res[, method := ifelse(cf_method == "ARF", paste(cf_method, n_synth), cf_method)]
res_mean <- res[, lapply(.SD, mean), .SDcols = eval_columns, by = .(method, dataset, id)]

res_log_probs_nondom = res[nondom == TRUE, lapply(.SD, mean), .SDcols = "log_probs", by = .(method, dataset, id)]
setnames(res_log_probs_nondom, "log_probs", "log_probs_nondom")

res_mean = merge(res_mean, res_log_probs_nondom, by = c("method", "dataset", "id"))

saveRDS(res, "R_experiments/res.Rds")
saveRDS(res_mean, "R_experiments/res_mean.Rds")

# Plot results -------------------------------------------------------------
# res_mean <- readRDS("R_experiments/res_mean.Rds")

plotdata = melt(res_mean, id.vars=c("method", "dataset", "id"))
plotdata_all = plotdata[!grepl("nondom", plotdata$variable),]
plotdata_nondom = plotdata[grepl("nondom", plotdata$variable),]

plot_results = function(data) {
  ggplot(data, aes(x = method, y = value, fill = method)) + 
    geom_boxplot() + 
    facet_grid(variable ~ dataset, scales = "free") + 
    theme_bw() +
    scale_color_brewer(palette="BrBG") + 
    theme(legend.position="none") 
}
plot1_all = plot_results(plotdata_all)
plot1_all
plot1_nondom = plot_results(plotdata_nondom)
plot1_nondom

ggsave(filename = "R_experiments/results_study_all.png", plot = plot1_all, dpi = 200, width = 6, height = 7)
ggsave(filename = "R_experiments/results_study_nondom.png", plot = plot1_nondom, dpi = 200, width = 6, height = 9)


# Time running ------------
# time_res = NULL
# registry = "registries/evaluate_simulation_22_02/"
# 
# loadRegistry(registry)
# done_ids = findDone()$job.id
# 
# for (id in done_ids) {
#   cols = c("job.id", "time.running")
#   time = unwrap(getJobTable(ids = id))[, ..cols]
#   exp_info = unwrap(getJobPars(id = id))
#   setnames(exp_info, "id", "dataset")
#   time_res = rbind(time_res, merge(time, exp_info, by = "job.id"))
# }
# time_res[, method := ifelse(cf_method == "ARF", paste(cf_method, n_synth), cf_method)]
# 
# plot2 = ggplot(time_res, aes(x = method, y = as.numeric(time.running), fill = method)) + 
#   geom_boxplot() + 
#   theme_bw() +
#   scale_color_brewer(palette="BrBG") + 
#   theme(legend.position="none") +
#   geom_line(aes(group=dataset), color = "gray")
# plot2
# 
# 
# ggsave(filename = "R_experiments/runtimes.png", plot = plot2, dpi = 200, width = 4, height = 4)
