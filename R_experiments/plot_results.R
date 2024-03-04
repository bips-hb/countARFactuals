# Some measures ------------------------------------------------------------
wilcox.test(res_mean$cor_lik, res_mean$cor_gow, alternative = "greater", paired = TRUE)
boxplot(cor_lik ~ method, res_mean)
boxplot(cor_gow ~ method, res_mean)
# komische Dinge bei NICE?

# boxplot(log(hv) ~ method, data = hvs)
# boxplot(log(hv_nondom) ~ method, data = hvs_nondom)

# Plot results -------------------------------------------------------------
res_mean <- readRDS("R_experiments/res_mean.Rds")
res_mean[, "log(runtime)" := log(runtime)]
res_mean[, "log(probs)" := log(log_probs)]
res_mean[, "log(probs_nondom)" := log(log_probs_nondom)]
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
p5 = plot_results(plotdata_all, "log(probs)")
p6 = plot_results(plotdata_nondom, "log(probs_nondom)", remove_strip_x = TRUE)
p7 = plot_results(plotdata_all, "log(runtime)", remove_strip_x = TRUE)
p8 = plot_results(plotdata_all, "number", remove_strip_x = TRUE)
p9 = plot_results(plotdata_nondom, "number_nondom", remove_strip_x = TRUE)
sell_plot = grid.arrange(p5, p6, p7, p8, p9, ncol = 1L)

ggsave(filename = "R_experiments/logprobs_runtime.png", plot = sell_plot, 
  dpi = 200, width = 10, height = 7)



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
