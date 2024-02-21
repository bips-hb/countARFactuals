# --- VISUALIZE RESULTS ----

library(batchtools)
library(ggplot2)
library(reshape2)


eval_columns = c("dist_x_interest", "no_changed", "dist_train", "dist_x_interest_all", "no_changed_all", "log_probs")

# Get results -------------------------------------------------------------

res = NULL

# registry1 = "registries/evaluate_simulation/"
# registry2 = "registries/evaluate_simulation_new/"
# 
# registries = c(registry1, registry2)
# 
# for (registry in c(registry1, registry2)) {
# 	loadRegistry(registry)
# 	done_ids = findDone()$job.id	
# 	
# 	for (id in done_ids) {
# 		cfe = readRDS(file.path(registry, "results", paste0(id, ".rds")))
# 		cols = c(eval_columns, "id")
# 		cfe = cfe[, ..cols]
# 		exp_info = unwrap(getJobPars(id = id))
# 		setnames(exp_info, "id", "dataset")
# 		res = rbind(res, cbind(cfe, exp_info))
# 	}
# }

files = list.files("cfs/")
files = files[grepl(files, pattern = "log_probs")]
for (file in files) {
	res = rbind(res, fread(file = file.path("cfs", "R", file), header = TRUE))
}


res[, method := ifelse(cf_method == "ARF", paste(cf_method, n_synth), cf_method)]

# cols <- c("diversity", "no_nondom", 
#   "frac_nondom", "hypervolume", "dist_x_interest_all", "no_changed_all", 
#   "neg_lik_all", "dist_x_interest_nondom", "no_changed_nondom", 
#   "neg_lik_nondom")
res_mean <- res[, lapply(.SD, mean), .SDcols = eval_columns, by = .(method, dataset, id)]

# saveRDS(res, "res.Rds")
# saveRDS(res_mean, "res_mean.Rds")

# Plot results -------------------------------------------------------------
# res_mean <- readRDS("res_mean.Rds")

plotdata = melt(res_mean, id.vars=c("method", "dataset", "id"))

ggplot(plotdata, aes(x = method, y = value, fill = method)) + 
	geom_boxplot() + 
    facet_grid(variable ~ dataset, scales = "free") + 
	theme_bw() +
	scale_color_brewer(palette="BrBG") + 
	theme(legend.position="none") 
