library(ggplot2)
library(reshape2)
library(scales)
library(patchwork)
library(data.table)

res_folder = file.path("R_experiments/results", Sys.Date())
dir.create(res_folder)

res_agg = readRDS("R_experiments/res_agg.Rds")

# Correlation analysis ------------------------------------------------------------
# FIXME: some correlations NA because all values constant (0 or 1)
# - 0 plausibility in bn_50_v2 (ARF, MOC), additionally two_sines, bn_5_v2 by NICE
# Removed those from evaluation

# Correlation estimated vs. true implausibility
corr_data = res_agg[!is.na(cor_lik) & !is.na(cor_gow),]
wilcox.test(corr_data$cor_lik, corr_data$cor_gow, alternative = "greater", paired = TRUE)
table(is.na(res_agg$cor_gow), res_agg$method)
boxplot(cor_lik ~ method, corr_data)
boxplot(cor_gow ~ method, corr_data)
# FIXME: weird things with NICE :/

# Plot results -------------------------------------------------------------
res = readRDS("R_experiments/res.Rds")
# only have a look on nondom! 
res = res[nondom == TRUE,]
res[, "plausibility" := exp(log_probs)]
res[, "sparsity" := 1 - rel_no_changed]
res[, "proximity" := 1 - dist_x_interest]

res_agg = readRDS("R_experiments/res_agg.Rds")
res_agg[, "runtime*" := log(runtime)]
res_agg[, "number*" := log(number)]
res_agg[, "hv*" := log(hv_normalized)]

res$dataset = factor(res$dataset, levels = c("cassini", "pawelczyk", 
  "two_sines", "bn_5_v2", "bn_10_v2", "bn_20", "bn_50_v2"), 
  labels = c("cassini", "pawelczyk", "two_sines", "bn_5", "bn_10", 
    "bn_20", "bn_50"))
res_agg$dataset = factor(res_agg$dataset, levels = c("cassini", "pawelczyk", 
  "two_sines", "bn_5_v2", "bn_10_v2", "bn_20", "bn_50_v2"), 
  labels = c("cassini", "pawelczyk", "two_sines", "bn_5", "bn_10", 
    "bn_20", "bn_50"))
res_agg$method = factor(res_agg$method, levels = c("NICE", "MOC", 
  "MOCCTREE", "MOCARF", "ARF"))
res$method = factor(res$method, levels = c("NICE", "MOC", 
  "MOCCTREE", "MOCARF", "ARF"))

## remove one outlier in NICE HV 
res_subset = res_agg[-which(res_agg$hv_normalized > 40 & res_agg$dataset == "bn_50"), ]

# ####  calculate ranks per objective ------
# res_mean = res_mean[!method %in% c("ARF 200"), 
#   c("rank_plausibility", "rank_proximity", "rank_sparsity", "rank_runtime") := lapply(.SD, frank, ties.method = "min"), 
#   .SDcols = c("plausibility", "proximity", "sparsity", "runtime"), 
#   by = .(dataset, id)]
# 
# res_mean = res_mean[!method %in% c("ARF 200"), 
#   c("rank_plausibility_nondom", "rank_proximity_nondom", "rank_sparsity_nondom") := lapply(.SD, frank, ties.method = "min"), 
#   .SDcols = c("plausibility_nondom", "proximity_nondom", "sparsity_nondom"), 
#   by = .(dataset, id)]
# 
# combine = function(x, mode = "mean") {
#   if (mode == "mean") {
#     m = round(mean(x), 2)
#     r = round(sd(x), 2) 
#   } else if (mode == "median") 
#   {
#     m = median(x)
#     r = IQR(x, na.rm = TRUE)
#   }
#   paste0(m, " [", r, "]")
# }
# 
# rank_mean = res_mean[, lapply(.SD, combine, mode = "mean"), 
#   .SDcols = c("rank_plausibility", "rank_proximity", "rank_sparsity", "rank_runtime"), 
#   by = .(method)]
# 
# rank_mean_nondom = res_mean[, lapply(.SD, combine, mode = "mean"), 
#   .SDcols = c("rank_plausibility_nondom", "rank_proximity_nondom", "rank_sparsity_nondom"), 
#   by = .(method)]
# 
# # print(xtable::xtable(rank_mean[!method %in% c("ARF 200")]), include.rownames = FALSE)
# print(xtable::xtable(rank_mean_nondom[!method %in% c("ARF 200")]), include.rownames = FALSE)


### get figures -----
# plotdata_agg = data.table(melt(res_agg, id.vars=c("method", "dataset", "id")))
# plotdata = data.table(melt(res, id.vars=c("method", "dataset", "id")))
# 
# boxplot(plausibility ~ method, data = res)

plot_results = function(data, evaluation_measure = NULL, remove_strip_x = FALSE) {
  if (!is.null(evaluation_measure)) {
    colnams = c("dataset", "method", "id", evaluation_measure)
    data = data[, ..colnams]
    setnames(data, evaluation_measure, "evalmeasure")
    data$measure = evaluation_measure
  }
  colours = c("ARF" = "deepskyblue3", "MOCARF" = "deepskyblue1", 
    "MOCCTREE" = "cornsilk4", "MOC" = "cornsilk3", "NICE" = "cornsilk2")
  pl = ggplot(data, aes(x = method, y = evalmeasure, fill = method)) + 
    geom_boxplot() + 
    coord_flip() +
    facet_grid(measure ~ dataset, scales = "free_x") + 
    theme_bw() + 
   #  scale_color_brewer(palette="BrBG") + 
    scale_fill_manual(values = colours) + 
    scale_y_continuous(breaks = pretty_breaks(n = 2)) +
    theme(legend.position="none", 
      axis.title.x = element_blank(), 
     axis.title.y = element_blank(), 
     panel.spacing.x = unit(4, "mm"),
     plot.margin=unit(c(.1, 0, 0, 0), "cm")) 
  if (remove_strip_x) {
    pl = pl + theme(strip.text.x = element_blank())
  }
  pl
  pl
}

# To show: 
p_plaus = plot_results(res, "plausibility")
p_plaus = plot_results(res, "log_probs")
p_prox = plot_results(res, "proximity", remove_strip_x = TRUE)
p_sparse = plot_results(res, "sparsity", remove_strip_x = TRUE)
p_hv = plot_results(res_subset, "hv_normalized", remove_strip_x = TRUE)
p_no = plot_results(res_agg, "number*", remove_strip_x = TRUE)
p_runtime = plot_results(res_agg, "runtime*", remove_strip_x = TRUE)

p_main = p_plaus / p_prox / p_sparse / p_hv / p_no / p_runtime
p_main

ggsave(filename = file.path(res_folder,"results_main.png"), plot = p_main, 
   dpi = 200, width = 8.5, height = 7)
