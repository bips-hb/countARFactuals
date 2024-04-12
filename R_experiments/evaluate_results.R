#--- EVALUATE RESULTS ---

library(ggplot2)
library(reshape2)
library(scales)
library(patchwork)
library(data.table)
library(eaf)

res_folder = file.path("R_experiments/results", Sys.Date())
dir.create(res_folder)

res_agg = readRDS("R_experiments/res_agg.Rds")

# Correlation analysis ------------------------------------------------------------
# Correlation estimated vs. true implausibility
corr_data = res_agg[!is.na(cor_lik) & !is.na(cor_gow) & dataset != "bn_50_v2",]
wilcox.test(corr_data$cor_lik, corr_data$cor_gow, alternative = "greater", paired = TRUE)
median(corr_data$cor_lik)
median(corr_data$cor_gow)
gcor = ggplot(corr_data, aes(x = cor_lik)) + 
  geom_density(aes(color = "o*plaus")) + 
  theme_bw() + 
  geom_density(aes(x = cor_gow, color = "o_plaus")) + 
  scale_colour_manual(name="plausibility",
    values=c("deepskyblue3", "cornsilk4")) + 
  xlab("correlation")
ggsave(filename = file.path(res_folder,"correlations.png"), plot = gcor, 
  dpi = 200, width = 5, height = 2)

# Plot results -------------------------------------------------------------
res = readRDS("R_experiments/res.Rds")
# only have a look on nondom! 
res = res[nondom == TRUE,]
res = res[dataset != "bn_50_v2"]
res[, "plausibility" := exp(log_probs)]
res[, "sparsity" := 1 - rel_no_changed]
res[, "proximity" := 1 - dist_x_interest]

res_agg = readRDS("R_experiments/res_agg.Rds")
res_agg[, "runtime*" := log(runtime)]
res_agg[, "number*" := log(number)]
res_agg[, "hypervolume" := hv_normalized]
res_agg[, "hv*" := log(hv_normalized)]

# Remove bn_50 due to weird hv results
res_agg = res_agg[dataset != "bn_50_v2", ]
res = res[dataset != "bn_50_v2", ]

# Scale hypervolume for plotting
scale01 = function(x){(x-min(x))/(max(x)-min(x))}
res_agg[, hypervolume := scale01(hypervolume), by = dataset]

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

res_subset = res_agg

# boxplot

plot_results = function(data, evaluation_measure = NULL, remove_strip_x = FALSE, 
                        log = FALSE) {
  if (!is.null(evaluation_measure)) {
    colnams = c("dataset", "method", "id", evaluation_measure)
    data = data[, ..colnams]
    setnames(data, evaluation_measure, "evalmeasure")
    data$measure = evaluation_measure
  }
  colours = c("ARF" = "deepskyblue3", "MOCARF" = "deepskyblue1", 
    "MOCCTREE" = "cornsilk4", "MOC" = "cornsilk3", "NICE" = "cornsilk2")
  pl = ggplot(data, aes(x = method, y = evalmeasure, fill = method)) + 
    geom_boxplot(outlier_size = 0.5) +
    coord_flip() +
    facet_grid(measure ~ dataset, scales = "free") + 
    theme_bw() + 
   #  scale_color_brewer(palette="BrBG") + 
    scale_fill_manual(values = colours) + 
    theme(legend.position="none", 
      axis.title.x = element_blank(), 
     axis.title.y = element_blank(), 
     panel.spacing.x = unit(4, "mm"),
     plot.margin=unit(c(.1, 0, 0, 0), "cm")) 
  if (remove_strip_x) {
    pl = pl + theme(strip.text.x = element_blank())
  }
  if (log) {
    pl = pl + scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x, n = 2))
  } else {
    pl = pl + scale_y_continuous(breaks = pretty_breaks(n = 2))
  }
  pl
}

res[, plausibility := log(plausibility)]
p_plaus = plot_results(res, "plausibility") 
p_prox = plot_results(res, "proximity", remove_strip_x = TRUE, log = FALSE)
p_sparse = plot_results(res, "sparsity", remove_strip_x = TRUE, log = FALSE) + 
  scale_y_continuous(breaks = c(0.1, 0.5, 0.9))
p_hv = plot_results(res_subset, "hypervolume", remove_strip_x = TRUE, log = TRUE) #+ 
  #scale_y_log10(breaks = c(1e-10, 1e-5, 1, 10))
p_no = plot_results(res_agg, "number", remove_strip_x = TRUE, log = TRUE) + 
  scale_y_log10(breaks = c(1, 10, 100))
p_runtime = plot_results(res_agg, "runtime", remove_strip_x = TRUE, log = TRUE)

p_main = p_plaus / p_prox / p_sparse / p_hv / p_no / p_runtime
p_main

ggsave(filename = file.path(res_folder,"results_main.png"), plot = p_main, 
   dpi = 200, width = 8.5, height = 7.5)


# compute empirical attainment functions

res[, implausibility := -exp(plausibility)]
res[, inproximity := -proximity]
res[, insparsity := -sparsity]

for (ds in unique(res$dataset)) {
  if (ds %in% c("pawelczyk", "two_sines")) {
    legend = "topright"
  } else {
    legend = "bottomleft"
  }
  xlim = NULL
  if (ds == "bn_10") {
    xlim = c(-0.002, 0.001)
  } else if (ds == "bn_20") {
    xlim = c(-4e-08, 4e-08)
  }
  pdf(file.path(res_folder, paste0("eafs_prox_", ds, ".pdf")), width = 4, height = 3.5)
  eafplot(implausibility + inproximity ~ id,
   groups = method, data=res[dataset == ds,], percentiles = c(50), ylab = "neg. proximity", 
    xlab = "neg. plausibility",
    legend.pos = legend, legend.txt = c("ARF", "MOC", "MOCARF", "MOCCTREE", "NICE"), xlim = xlim)
  dev.off()
  
  xlim = NULL
  legend = "bottomleft"
  if (ds == "bn_10") {
    xlim = c(-0.0015, 0)
  } else if (ds == "bn_20") {
    xlim = c(-2e-8, 0)
  } else if (ds == "cassini") {
    xlim = c(-3.1, 0)
  } else if (ds == "two_sines")  {
    xlim = c(-0.36, 0)
  } else if (ds == "pawelczyk") {
    xlim = c(-0.036, 0)
  }
  pdf(file.path(res_folder, paste0("eafs_spars_", ds, ".pdf")), width = 4, height = 3.5)
  eafplot(implausibility + insparsity ~ id,
    groups = method, data=res[dataset == ds,], percentiles = c(50), ylab = "neg. sparsity", 
    xlab = "neg. plausibility",
    legend.pos = legend, legend.txt = c("ARF", "MOC", "MOCARF", "MOCCTREE", "NICE"), xlim = xlim)
  dev.off()
}


