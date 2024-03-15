######## SAVE CFEs
library(batchtools)

outdir = "cfs/test"
dir.create(outdir)
registry = "test"


loadRegistry(registry)
done_ids = findDone()$job.id

for (id in done_ids) {
	cfe = readRDS(file.path(registry, "results", paste0(id, ".rds")))
	exp_info = unwrap(getJobPars(id = id))
	setnames(exp_info, "id", "dataset")
	cfe = cbind(cfe, exp_info)
	write.csv(cfe, row.names = FALSE, file.path(outdir,
												paste0(paste(exp_info$dataset, exp_info$cf_method, exp_info$n_synth, 
															 exp_info$node_selector, sep = "_"), ".csv")))
}


