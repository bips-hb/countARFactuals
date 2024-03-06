library(ggplot2)
library(reshape2)
library(scales)
library(patchwork)
library(data.table)

res_folder = file.path("R_experiments/results", Sys.Date())
dir.create(res_folder)

res_mean = readRDS("R_experiments/res_mean.Rds")
res_mean = res_mean[method != "WhatIf", ]

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
res_mean[, "runtime" := log(runtime)]
res_mean[, "log(probs)" := log(log_probs)]
res_mean[, "plausibility" := -log(log_probs)]
res_mean[, "plausibility_nondom" := -log(log_probs_nondom)]
res_mean[, "log(probs_nondom)" := log(log_probs_nondom)]
res_mean[, "log(hv)" := log(hv)]
res_mean[, "log(hv_nondom)" := log(hv_nondom)]
res_mean[, "no" := log(number)]
res_mean[, "no_nondom" := log(number_nondom)]
res_mean$dataset = factor(res_mean$dataset, levels = c("cassini", "pawelczyk", 
  "two_sines", "bn_5_v2", "bn_10_v2", "bn_20", "bn_50_v2"), 
  labels = c("cassini", "pawelczyk", "two_sines", "bn_5", "bn_10", 
    "bn_20", "bn_50"))
res_mean$method = factor(res_mean$method, levels = c("WhatIf", "NICE", "MOC", 
  "MOCCTREE", "MOCARF", "ARF 20", "ARF 200"))

res_mean[method == "ARF 20", method := "ARF"]
res_mean[, method := droplevels(method)]

# rename objectives
setnames(res_mean, old = c("dist_x_interest", "no_changed", 
  "dist_x_interest_nondom", "no_changed_nondom"), 
  new = c("proximity", "sparsity", "proximity_nondom", "sparsity_nondom"))



####  calculate ranks per objective ------
res_mean = res_mean[!method %in% c("ARF 200"), 
  c("rank_plausibility", "rank_proximity", "rank_sparsity", "rank_runtime") := lapply(.SD, frank, ties.method = "min"), 
  .SDcols = c("plausibility", "proximity", "sparsity", "runtime"), 
  by = .(dataset, id)]

res_mean = res_mean[!method %in% c("ARF 200"), 
  c("rank_plausibility_nondom", "rank_proximity_nondom", "rank_sparsity_nondom") := lapply(.SD, frank, ties.method = "min"), 
  .SDcols = c("plausibility_nondom", "proximity_nondom", "sparsity_nondom"), 
  by = .(dataset, id)]

combine = function(x, mode = "mean") {
  if (mode == "mean") {
    m = round(mean(x), 2)
    r = round(sd(x), 2) 
  } else if (mode == "median") 
  {
    m = median(x)
    r = IQR(x, na.rm = TRUE)
  }
  paste0(m, " [", r, "]")
}

rank_mean = res_mean[, lapply(.SD, combine, mode = "mean"), 
  .SDcols = c("rank_plausibility", "rank_proximity", "rank_sparsity", "rank_runtime"), 
  by = .(method)]

rank_mean_nondom = res_mean[, lapply(.SD, combine, mode = "mean"), 
  .SDcols = c("rank_plausibility_nondom", "rank_proximity_nondom", "rank_sparsity_nondom"), 
  by = .(method)]

print(xtable::xtable(rank_mean[!method %in% c("ARF 200")]), include.rownames = FALSE)
print(xtable::xtable(rank_mean_nondom[!method %in% c("ARF 200")]), include.rownames = FALSE)


### get figures -----
plotdata = data.table(melt(res_mean, id.vars=c("method", "dataset", "id")))
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
    scale_y_continuous(breaks = pretty_breaks(n = 2)) + 
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

# To show: 
p_plaus = plot_results(plotdata[method != "ARF 200"], "plausibility")
p_prox = plot_results(plotdata[method != "ARF 200"], "proximity", remove_strip_x = TRUE)
p_sparse = plot_results(plotdata[method != "ARF 200"], "sparsity", remove_strip_x = TRUE)
p_runtime = plot_results(plotdata[method != "ARF 200"], "runtime", remove_strip_x = TRUE)

p_main = p_plaus / p_prox / p_sparse / p_runtime
p_main

ggsave(filename = file.path(res_folder,"results_main.png"), plot = p_main, 
   dpi = 200, width = 8, height = 4)


# Appendix 
p_hv = plot_results(plotdata[method != "ARF 200"], "hv", remove_strip_x = TRUE)
p_no = plot_results(plotdata[method != "ARF 200"], "number", remove_strip_x = TRUE)

p_hv_no = p_hv / p_no

ggsave(filename = file.path(res_folder,"hv_no.png"), plot = p_hv_no, 
  dpi = 200, width = 8, height = 2)

p_plaus_nd = plot_results(plotdata_nondom[method != "ARF 200"], "plausibility_nondom")
p_prox_nd = plot_results(plotdata_nondom[method != "ARF 200"], "proximity_nondom", remove_strip_x = TRUE)
p_sparse_nd = plot_results(plotdata_nondom[method != "ARF 200"], "sparsity_nondom", remove_strip_x = TRUE)
p_hv_nd = plot_results(plotdata_nondom[method != "ARF 200"], "hv_nondom", remove_strip_x = TRUE)
p_no_nondom = plot_results(plotdata_nondom[method != "ARF 200"], "number_nondom", remove_strip_x = TRUE)

p_main_nd = p_plaus_nd / p_prox_nd / p_sparse_nd / p_hv_nd / p_no_nondom
p_main_nd

ggsave(filename = file.path(res_folder,"results_main_nondom.png"), plot = p_main_nd, 
  dpi = 200, width = 8, height = 8)

# p_runtime2 = plot_results(plotdata_all[method %in% c("ARF", "ARF 200")], "runtime", remove_strip_x = FALSE)
# ggsave(filename = file.path(res_folder,"runtimes_ARF.png"), plot = p_runtime2, 
#   dpi = 200, width = 8, height = 1)
