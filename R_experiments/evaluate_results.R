library(ggplot2)
library(reshape2)
library(scales)
library(patchwork)
library(data.table)

res_folder = file.path("R_experiments/results", Sys.Date())
dir.create(res_folder)

res_mean = readRDS("R_experiments/res_mean.Rds")

# Correlation analysis ------------------------------------------------------------
# FIXME: some correlations NA because all values constant (0 or 1)
# - 0 plausibility in bn_50_v2 (ARF, MOC), additionally two_sines, bn_5_v2 by NICE
# Removed those from evaluation --> all of Whatif, some of the others
corr_data = res_mean[!is.na(cor_lik) & !is.na(cor_gow),]
wilcox.test(corr_data$cor_lik, corr_data$cor_gow, alternative = "greater", paired = TRUE)
table(is.na(res_mean$cor_gow), res_mean$method)

# Correlation
boxplot(cor_lik ~ method, corr_data)
boxplot(cor_gow ~ method, corr_data)
# FIXME: weird things with NICE :/

# Plot results -------------------------------------------------------------
res_mean[, "log(runtime)" := log(runtime)]
res_mean[, "log(probs)" := log(log_probs)]
res_mean[, "log(probs_nondom)" := log(log_probs_nondom)]
res_mean[, "log(hv)" := log(hv)]
res_mean[, "log(hv_nondom)" := log(hv_nondom)]
res_mean[, "log(no)" := log(number)]
res_mean[, "log(no_nondom)" := log(number_nondom)]
res_mean$dataset = factor(res_mean$dataset, levels = c("cassini", "pawelczyk", 
  "two_sines", "bn_5_v2", "bn_10_v2", "bn_20", "bn_50_v2"), 
  labels = c("cassini", "pawelczyk", "two_sines", "bn_5", "bn_10", 
    "bn_20", "bn_50"))
res_mean$method = factor(res_mean$method, levels = c("WhatIf", "NICE", "MOC", 
  "MOCCTREE", "MOCARF", "ARF 20", "ARF 200"))

# rename objectives
setnames(res_mean, old = c("dist_x_interest", "no_changed", "dist_train", 
  "dist_x_interest_nondom", "no_changed_nondom", "dist_train_nondom"), 
  new = c("o_prox", "o_sparse", "o_plaus", "o_prox_nondom", "o_sparse_nondom", 
    "o_plaus_nondom"))

plotdata = data.table(melt(res_mean, id.vars=c("method", "dataset", "id")))
plotdata_all = plotdata[!grepl("nondom", plotdata$variable),]
plotdata_nondom = plotdata[grepl("nondom", plotdata$variable),]

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
    scale_y_continuous(breaks = pretty_breaks(n = 3)) + 
    theme(legend.position="none", 
      axis.title.x = element_blank(), 
      axis.title.y = element_blank(), 
      panel.spacing.x = unit(4, "mm"),
      plot.margin=unit(c(0, 0, 0, 0), "cm")) 
  if (remove_strip_x) {
    pl = pl + theme(strip.text.x = element_blank())
  }
  pl
  pl
  
}

# objectives
p_prox = plot_results(plotdata_all[method != "ARF 200"], "o_prox", remove_strip_x = TRUE)
p_sparse = plot_results(plotdata_all[method != "ARF 200"], "o_sparse", remove_strip_x = TRUE)
p_plaus = plot_results(plotdata_all[method != "ARF 200"], "o_plaus", remove_strip_x = TRUE)
p_neglik = plot_results(plotdata_all[method != "ARF 200"], "neg_lik", remove_strip_x = TRUE)
obj_plot = p_prox/p_sparse/p_plaus/p_neglik

# ggsave(filename = file.path(res_folder,"results_objectives.png"), plot = obj_plot, 
#   dpi = 200, width = 11, height = 6)

# nondom 
p_prox_nondom = plot_results(plotdata_nondom, "o_prox_nondom", remove_strip_x = TRUE)
p_sparse_nondom = plot_results(plotdata_nondom, "o_sparse_nondom", remove_strip_x = TRUE)
p_plaus_nondom = plot_results(plotdata_nondom, "o_plaus_nondom", remove_strip_x = TRUE)
p_neglik_nondom = plot_results(plotdata_nondom, "neg_lik_nondom", remove_strip_x = TRUE)

obj_plot_nondom = p_prox_nondom/ p_sparse_nondom/ p_plaus_nondom/ p_neglik_nondom
# ggsave(filename = file.path(res_folder,"results_objectives_nondom.png"), plot = obj_plot_nondom, 
#   dpi = 200, width = 11, height = 6)

# selling points 
p_probs = plot_results(plotdata_all[method != "ARF 200"], "log(probs)")
p_probs_nondom = plot_results(plotdata_nondom[method != "ARF 200"], "log(probs_nondom)", remove_strip_x = TRUE)
p_hv = plot_results(plotdata[method != "ARF 200"], "hv", remove_strip_x = TRUE)
p_hv_nondom = plot_results(plotdata_nondom[method != "ARF 200"], "hv_nondom", remove_strip_x = TRUE)

# sell_obj = p_probs/p_probs_nondom/p_hv/p_hv_nondom
# ggsave(filename = file.path(res_folder,"logprobs_hv.png"), plot = sell_obj, 
#   dpi = 200, width = 10, height = 6)

p_runtime = plot_results(plotdata_all[method != "ARF 200"], "log(runtime)", remove_strip_x = TRUE)
p_no = plot_results(plotdata_all[method != "ARF 200"], "log(no)", remove_strip_x = TRUE)
p_no_nondom = plot_results(plotdata_nondom[method != "ARF 200"], "log(no_nondom)", remove_strip_x = TRUE)

# sell_time_no = p_runtime/p_no/p_no_nondom
# ggsave(filename = file.path(res_folder,"runtime_no.png"), plot = sell_time_no, 
#   dpi = 200, width = 10, height = 5)

# to show: 
p_obj_hv = p_probs / p_sparse /p_prox / p_hv /p_runtime /p_no
ggsave(filename = file.path(res_folder,"obj_hv.png"), plot = p_obj_hv, 
  dpi = 200, width = 8, height = 6.5)

p_obj_hv_nondom = p_probs_nondom / p_sparse_nondom /p_prox_nondom / p_hv_nondom / p_no_nondom
ggsave(filename = file.path(res_folder,"obj_hv_nondom.png"), plot = p_obj_hv_nondom, 
  dpi = 200, width = 8, height = 7)

p_runtime2 = plot_results(plotdata_all[method %in% c("ARF 20", "ARF 200")], "log(runtime)", remove_strip_x = FALSE)
ggsave(filename = file.path(res_folder,"runtimes_ARF.png"), plot = p_runtime2, 
  dpi = 200, width = 8, height = 2)

