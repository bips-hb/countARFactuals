# --- VISUALIZE RESULTS ----
library(data.table)
library(batchtools)
library(ggplot2)
library(reshape2)
library(gridExtra)


eval_columns = c("dist_x_interest", "no_changed", "neg_lik", "dist_train", 
  "dist_x_interest_nondom", "no_changed_nondom", "neg_lik_nondom", 
  "dist_train_nondom", "log_probs", "runtime")
save_columns = c(eval_columns, "id", "job.id", "n_synth", "problem", "cf_method", "dataset", "nondom")
csv_dir = "cfs/26_02"

# Get results -------------------------------------------------------------

res = NULL

files = list.files(csv_dir)
files = files[grepl(files, pattern = "log_probs.csv")]
for (file in files) {
  print(file)
  tmp = fread(file = file.path(csv_dir, file), header = TRUE)[, ..save_columns]
  res = rbind(res, tmp)
}

#### FIXME #####
res = res[, log_probs := exp(log_probs)]
# ~2000
###################
res[, method := ifelse(cf_method == "ARF", paste(cf_method, n_synth), cf_method)]

# res = res[, plausibility := 1-log_probs]
# eval_measures = c("dist_x_interest", "no_changed", "plausibility")
# hv_dat = copy(res)[, c(eval_measures, "method", "dataset", "id"), with = FALSE]
# 
# get_hv = function(x) {
# 	if (nrow(x) > 0) {
#         res = miesmuschel::domhv(fitnesses = -as.matrix(rbind(x)),
#           nadir = -c(1, 1, 1)
#         )
# 	} else {res = -Inf}
#     return(res)
# }
# 
# hv = hv_dat %>% 
#       group_by(method, dataset, id) %>%
#       group_modify(~ data.frame(cbind(.x, "dom_hv" = get_hv(.x))))
# hv_dat[, hv := get_hv(..eval_measures), by = .(method, dataset, id)]

res_mean <- res[, lapply(.SD, mean), .SDcols = eval_columns, by = .(method, dataset, id)]

res_log_probs_nondom = res[nondom == TRUE, lapply(.SD, mean), .SDcols = "log_probs", by = .(method, dataset, id)]
setnames(res_log_probs_nondom, "log_probs", "log_probs_nondom")

res_mean = merge(res_mean, res_log_probs_nondom, by = c("method", "dataset", "id"))


# saveRDS(res, "R_experiments/res.Rds")
# saveRDS(res_mean, "R_experiments/res_mean.Rds")

# Plot results -------------------------------------------------------------
res_mean <- readRDS("R_experiments/res_mean.Rds")
res_mean[, "log(runtime)" := log(runtime)]
res_mean$dataset = factor(res_mean$dataset, levels = c("cassini", "pawelczyk", "two_sines", "bn_5_v2", "bn_10_v2", "bn_20", "bn_50_v2"))

setnames(res_mean, old = c("dist_x_interest", "no_changed", "dist_train", "dist_x_interest_nondom", "no_changed_nondom", "dist_train_nondom"), 
    new = c("o_prox", "o_sparse", "o_plaus", "o_prox_nondom", "o_sparse_nondom", "o_plaus_nondom"))

plotdata = data.table(melt(res_mean, id.vars=c("method", "dataset", "id")))
plotdata_all = plotdata[!grepl("nondom", plotdata$variable),]
plotdata_nondom = plotdata[grepl("nondom", plotdata$variable),]

# label_names <- c(
#   "runtime" = "runtime",
#   "log(runtime)" = "log(runtime)",
#   "dist_x_interest" = "test",
#   "no_changed" = expression(o[sparse]),
#   "dist_train" = expression(o[plaus]),
#   "log_probs" = "log(probs)"
# )

plot_results = function(data, evaluation_measures = NULL, remove_strip_x = FALSE) {
  if (!is.null(evaluation_measures)) {
    data = data[variable %in% evaluation_measures, ]
  }
 
  pl = ggplot(data, aes(x = method, y = value, fill = method)) + 
    geom_boxplot() + 
    coord_flip() +
    facet_grid(variable ~ dataset, scales = "free_x") + 
    theme_bw() +
    scale_color_brewer(palette="BrBG") + 
    # scale_y_continuous(breaks = pretty_breaks(n = 3)) + 
    theme(legend.position="none", 
      axis.title.x = element_blank(), 
      axis.title.y = element_blank(), 
      plot.margin=unit(c(0, 0, 0, 0), "cm")) 
  if (remove_strip_x) {
    pl = pl + theme(strip.text.x = element_blank())
  }
  pl
  pl
  
}

# objectives
p1 = plot_results(plotdata_all, "o_prox")
p2 = plot_results(plotdata_all, "o_sparse", remove_strip_x = TRUE)
p3 = plot_results(plotdata_all, "o_plaus", remove_strip_x = TRUE)
p4 = plot_results(plotdata_all, "neg_lik", remove_strip_x = TRUE)
obj_plot = grid.arrange(p1, p2, p3, p4, ncol = 1L)

ggsave(filename = "R_experiments/results_objectives.png", plot = obj_plot, 
  dpi = 200, width = 11, height = 6)

# nondom 
p1 = plot_results(plotdata_nondom, "o_prox_nondom")
p2 = plot_results(plotdata_nondom, "o_sparse_nondom", remove_strip_x = TRUE)
p3 = plot_results(plotdata_nondom, "o_plaus_nondom", remove_strip_x = TRUE)
p4 = plot_results(plotdata_nondom, "neg_lik_nondom", remove_strip_x = TRUE)
obj_plot_nondom = grid.arrange(p1, p2, p3, p4, ncol = 1L)

ggsave(filename = "R_experiments/results_objectives_nondom.png", plot = obj_plot_nondom, 
  dpi = 200, width = 11, height = 7.5)

# selling points 
p5 = plot_results(plotdata_all, "log_probs")
p6 = plot_results(plotdata_nondom, "log_probs_nondom", remove_strip_x = TRUE)
p7 = plot_results(plotdata_all, "log(runtime)", remove_strip_x = TRUE)
sell_plot = grid.arrange(p5, p6, p7, ncol = 1L)

ggsave(filename = "R_experiments/logprobs_runtime.png", plot = sell_plot, 
  dpi = 200, width = 10, height = 4.5)



# ggsave(filename = "R_experiments/results_study_all.png", plot = plot1_all, 
#   dpi = 200, width = 15, height = 7)
# ggsave(filename = "R_experiments/results_study_nondom.png", plot = plot1_nondom, 
#   dpi = 200, width = 15, height = 9)


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
