# --- VISUALIZE RESULTS ----
library(data.table)
library(batchtools)
library(ggplot2)
library(reshape2)


eval_columns = c("dist_x_interest", "no_changed", "neg_lik", "dist_train", "dist_x_interest_all", "no_changed_all", "log_probs")

csv_dir = "cfs/23_02"

# Get results -------------------------------------------------------------

res = NULL



files = list.files(csv_dir)
files = files[grepl(files, pattern = "log_probs")]
for (file in files) {
	res = rbind(res, fread(file = file.path(csv_dir, file), header = TRUE))
}


res[, method := ifelse(cf_method == "ARF", paste(cf_method, n_synth), cf_method)]

# cols <- c("diversity", "no_nondom", 
#   "frac_nondom", "hypervolume", "dist_x_interest_all", "no_changed_all", 
#   "neg_lik_all", "dist_x_interest_nondom", "no_changed_nondom", 
#   "neg_lik_nondom")
res_mean <- res[, lapply(.SD, mean), .SDcols = eval_columns, by = .(method, dataset, id)]

saveRDS(res, "R_experiments/res.Rds")
saveRDS(res_mean, "R_experiments/res_mean.Rds")

# Plot results -------------------------------------------------------------
# res_mean <- readRDS("res_mean.Rds")

plotdata = melt(res_mean, id.vars=c("method", "dataset", "id"))

plot1 = ggplot(plotdata, aes(x = method, y = value, fill = method)) + 
	geom_boxplot() + 
    facet_grid(variable ~ dataset, scales = "free") + 
	theme_bw() +
	scale_color_brewer(palette="BrBG") + 
	theme(legend.position="none") 


ggsave(filename = "R_experiments/results_study.png", plot = plot1, dpi = 200, width = 8, height = 8)


# Time running ------------
time_res = NULL
registry1 = "registries/evaluate_simulation/"
registry2 = "registries/evaluate_simulation_new/"

registries = c(registry1, registry2)

for (registry in c(registry1, registry2)) {
	loadRegistry(registry)
	done_ids = findDone()$job.id

	for (id in done_ids) {
		cols = c("job.id", "time.running")
		time = unwrap(getJobTable(ids = id))[, ..cols]
		exp_info = unwrap(getJobPars(id = id))
		setnames(exp_info, "id", "dataset")
		time_res = rbind(time_res, merge(time, exp_info, by = "job.id"))
	}
}

time_res[, method := ifelse(cf_method == "ARF", paste(cf_method, n_synth), cf_method)]

plot2 = ggplot(time_res, aes(x = method, y = as.numeric(time.running), fill = method)) + 
	geom_boxplot() + 
	theme_bw() +
	scale_color_brewer(palette="BrBG") + 
	theme(legend.position="none") +
	geom_line(aes(group=dataset), color = "gray")
plot2


ggsave(filename = "R_experiments/runtimes.png", plot = plot2, dpi = 200, width = 4, height = 4)
