######## SAVE CFEs
library(batchtools)

outdir = "cfs"

registry1 = "registries/evaluate_simulation/"
registry2 = "registries/evaluate_simulation_new/"

for (registry in c(registry1, registry2)) {
	loadRegistry(registry)
	done_ids = findDone()$job.id	
	
	for (id in done_ids) {
		cfe = readRDS(file.path(registry1, "results", paste0(id, ".rds")))
		# exp_info = unwrap(getJobPars(id = id))
		# extra_columns = c("dist_x_interest", "no_changed", "dist_train", "dist_x_interest_all", "no_changed_all", "id")
		# cfe = cfe[, !..extra_columns]
		write.csv(cfe, row.names = FALSE, file.path(outdir,
								 paste0(paste(exp_info$id, exp_info$cf_method, exp_info$n_synth, 
								 			 exp_info$node_selector, sep = "_"), ".csv")))
	}
}

