######## SAVE CFEs
library(batchtools)

outdir = "cfs/26_02"

registry = "registries/evaluate_simulation_23_02/"


loadRegistry(registry)
done_ids = findDone()$job.id

for (id in done_ids) {
	cfe = readRDS(file.path(registry, "results", paste0(id, ".rds")))
	exp_info = unwrap(getJobPars(id = id))
	setnames(exp_info, "id", "dataset")
	cfe = cbind(cfe, exp_info)
	# extra_columns = c("dist_x_interest", "no_changed", "dist_train", "dist_x_interest_all", "no_changed_all", "id")
	# cfe = cfe[, !..extra_columns]
	write.csv(cfe, row.names = FALSE, file.path(outdir,
												paste0(paste(exp_info$dataset, exp_info$cf_method, exp_info$n_synth, 
															 exp_info$node_selector, sep = "_"), ".csv")))
}


