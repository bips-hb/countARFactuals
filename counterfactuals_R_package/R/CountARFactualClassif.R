#' CountARFactual (ARF-based Counterfactuals) for Classification Tasks
#' 
#' @description 
#' CountARFactuals generates counterfactuals based on the adversarial random forests introduced in Watson et. al 2023
#' 
#' @references 
#'
#' Watson, D. S., Blesch, K., Kapar, J., and Wright, M. N. (2023). Adversarial
#' random forests for density estimation and generative modeling. In Proceedings
#' of the 26th International Conference on Artificial Intelligence and Statistics, 
#' pages 5357–5375. PMLR.
#' 
#' @examples 
#' if (require("randomForest")) {
#'  \donttest{
#'   # Train a model
#'   rf = randomForest(Species ~ ., data = iris)
#'   # Create a predictor object
#'   predictor = iml::Predictor$new(rf, type = "prob")
#'   # Find counterfactuals for x_interest
#'   arf_classif = CountARFactualClassif$new(predictor)
#'  
#'   cfactuals = arf_classif$find_counterfactuals(
#'     x_interest = iris[150L, ], desired_class = "versicolor", desired_prob = c(0.5, 1)
#'   )
#'   # Print the counterfactuals
#'   cfactuals$data
#'   # Plot evolution of hypervolume and mean and minimum objective values
#'   cfactuals$evaluate_set()
#'   cfactuals$plot_parallel()
#'   }
#' }
#' 
#' @export
CountARFactualClassif = R6::R6Class("CountARFactualClassif", 
  inherit = CounterfactualMethodClassif,
  
  public = list(
    #' @description Create a new `MOCClassif` object.
    #' @template predictor
    #' @param max_feats_to_change (`numeric(1)`)\cr  
    #' The maximum number of features allowed to be altered (default number of features of `predictor$data`).
    #' If max_feats_to_change is larger the number of features after account for fixed features, a message is printed. 
    #' @param n_synth (`numeric(1)`) \cr 
    #' The number of samples drawn from the marginal distributions (default 10L).
    #' @param n_iterations (`numeric(1)`) \cr 
    #' The number of iterations. In each iteration a new terminal node is chosen 
    #' from which the `n_synth` candidates are drawn (default 50L).
    #' @param feature_selector (`character(1)`)\cr
    #' The method to choose features that are fixed for a counterfactual (and thus 
    #' part of the conditioning set when choosing tree paths). The default 
    #' `random_importance` means that the probability of being in the conditioning set 
    #' is proportional to how unimportant a feature is (unimportant features are more likely to be fixed). 
    #' The strategy `importance` means that all features are fixed except for 
    #' the up to `max_feats_to_change` most important features. 
    #' The strategy `random` randomly chooses up to `max_feats_to_change` 
    #' features that are not part of the conditioning set. All others are part.
    #' @param importance_method (`character(1)`)\cr 
    #' The local importance method used to choose variables for the conditioning set. 
    #' Ignored if `feature_selector = "random"`. 
    #' Either "fastshap" based on the `fastshap` package or "icesd" (default)
    #' based on the standard deviation of the ICE curve is possible.
    #' @param fixed_features (`character()`|`NULL`)\cr
    #' Names of features that are not allowed to be changed. NULL (default) allows all features to be changed.
    #' @param node_selector (`character(1)`)\cr
    #' How to select a node, based on "coverage" alone (default) or on coverage and proximity ("coverage_proximity").
    #' @param weight_node_selector (`character(1)`) \cr
    #' How to weight coverage and proximity when `node_selector` is "coverage_proximity".
    #' @param arf (`ranger`) \cr
    #'   Fitted arf. If NULL, arf is newly fitted. 
    #' @param psi (`list`) \cr
    #'   Fitted forde object. If NULL, arf::forde is called. 
    #' 
    #' @export
    initialize = function(predictor, max_feats_to_change = predictor$data$n.features, 
      n_synth = 20L, n_iterations = 50L, feature_selector = "random_importance", 
      importance_method = "icesd", node_selector = "coverage", 
      weight_node_selector = c(20, 20), fixed_features = NULL, arf = NULL, psi = NULL) { 
      # TODO: add other hyperparameter
      super$initialize(predictor)
      checkmate::assert_integerish(max_feats_to_change, lower = 1L, upper = predictor$data$n.features)
      checkmate::assert_integerish(n_synth, lower = 1L)
      checkmate::assert_choice(feature_selector, choices = c("importance", "random", "random_importance"))
      checkmate::assert_choice(importance_method, choices = c("fastshap", "icesd"))
      if (!is.null(fixed_features)) {
        assert_names(fixed_features, subset.of = private$predictor$data$feature.names)
      }
      checkmate::assert_class(arf, "ranger", null.ok = TRUE)
    
      private$max_feats_to_change = max_feats_to_change
      private$feature_selector = feature_selector
      private$max_feats_to_change = max_feats_to_change
      private$n_synth = n_synth
      private$n_iterations = n_iterations
      private$importance_method = importance_method
      private$arf = arf
      private$psi = psi
      private$fixed_features = fixed_features
      private$node_selector = node_selector
      private$weight_node_selector = weight_node_selector
    }
  ),
  active = list(
    #' @field arf_iterations (`numeric(1)`)\cr
    #' The number of iterations for the arf to terminate.
    arf_iterations = function(value) {
      if (missing(value)) {
        private$.arf_iterations
      } else {
        stop("`$arf_iterations` is read only", call. = FALSE)
      }
    }
  ),
  private = list(
    max_feats_to_change = NULL,
    n_synth = NULL,
    n_iterations = NULL,
    feature_selector = NULL,
    importance_method = NULL,
    fixed_features = NULL,
    .arf_iterations = NULL,
    arf = NULL,
    psi = NULL,
    node_selector = NULL,
    weight_node_selector = NULL,
    run = function() {
      
      # Fit ARF
      dat = copy(private$predictor$data$get.x())
      dat[, yhat := private$predictor$predict(dat)[,private$desired_class]]
      if (is.null(private$arf)) {
        private$arf = adversarial_rf(dat, always.split.variables = "yhat")
      }
      private$.arf_iterations = length(private$arf$acc)
      if (is.null(private$psi)) {
        psi = forde(private$arf, dat)
      } else {
        psi = private$psi
      }
      # Gower distances
      leaf_means = dcast(psi$cnt[variable != "yhat", .(f_idx, variable, mu)], f_idx ~ variable, value.var = "mu")
      leaf_dist = data.table(f_idx = leaf_means$f_idx, dist = gower:::gower_dist(leaf_means, private$x_interest))
      
      if (private$node_selector == "coverage_proximity") {
        # Use weighted combination of coverage and weights as new leaf weights
        psi$forest = merge(psi$forest, leaf_dist, by = "f_idx")
        weight_cvg = private$weight_node_selector[1]
        weight_dist = private$weight_node_selector[2]
        psi$forest[, cvg := exp(weight_cvg*cvg-weight_dist*dist)]
        psi$forest[, dist := NULL]
      }
      
      flex_cols = setdiff(names(private$x_interest), private$fixed_features)
      
      # Conditional sampling
      ##  Select fixed variables/conditioning set
      if (grepl("importance", private$feature_selector)) {
        # Shapley values as local importance
        if (private$importance_method == "fastshap") {
          if (!requireNamespace("fastshap", quietly = TRUE)) {
            stop("Package 'fastshap' needed for this measuring importance. Please install it.", call. = FALSE)
          }
          pfun = function(object, newdata) { 
            unname(object$predict(newdata)[,private$desired_class])
          }
          shap = fastshap::explain(private$predictor, X = private$predictor$data$get.x(), 
            pred_wrapper = pfun, newdata = private$x_interest, feature_names = flex_cols,
            nsim = 1000)
          vim = abs(shap[1, ])
          # ICE curve standard deviation as local importance
        } else if (private$importance_method == "icesd") {
          param_set = make_param_set(private$predictor$data$get.x())
          vim = get_ICE_sd(private$x_interest, private$predictor, param_set)
          vim = vim[flex_cols]
        }
      }
      
      evidence = rbindlist(lapply(1:private$n_iterations, function(i) {
        max_possible = private$predictor$data$n.features - length(private$fixed_features)
        if (max_possible >= private$max_feats_to_change) {
          max_possible = private$max_feats_to_change
        }
        feats_not_to_change = sample(seq(
          private$predictor$data$n.features - max_possible - length(private$fixed_features), 
          private$predictor$data$n.features - 1 - length(private$fixed_features)), size = 1L)
        if (private$feature_selector == "importance") {
          ordered_features = names(sort(vim)) # Smallest (= less important) first
          cols = ordered_features[1:feats_not_to_change]
        } else if (private$feature_selector == "random_importance") {
          # assert_true(all(names(vim) == private$predictor$data$feature.names))
          # get probabilities using softmax
          softmax = function(x) exp(x)/sum(exp(x))
          p_selected = softmax(-vim)
          cols = sample(flex_cols, 
            size = feats_not_to_change,
            prob = p_selected)
        } else if (private$feature_selector == "random") {
          cols = sample(flex_cols, 
            size = feats_not_to_change)
        }
        cols = c(cols, private$fixed_features)
        fixed = copy(private$x_interest)
        na_cols = setdiff(colnames(fixed), cols)
        fixed[, (na_cols) := NA]
        evidence = fixed
        evidence = data.table(evidence, 
                              yhat = paste0("(", min(private$desired_prob), ",", 
                                            max(private$desired_prob), ")"))
      }))
      synth = forge(psi, n_synth = private$n_synth, condition = evidence)
      synth[, yhat := NULL]
      
      # Recode factors to original factor levels
      factor_cols = names(which(sapply(private$predictor$data$X, is.factor)))
      for (factor_col in factor_cols) {
        fact_col_pred = private$predictor$data$X[[factor_col]]
        value =  factor(synth[[factor_col]], levels = levels(fact_col_pred), 
          ordered = is.ordered(fact_col_pred))
        set(synth, j = factor_col, value = value)
      }
      
      # Keep only valid counterfactuals
      # cfs = synth[between(private$predictor$predict(synth)[,private$desired_class], 
      #   private$desired_prob[1L], private$desired_prob[2L]),]
      unique(synth)
    },
    
    print_parameters = function() {
      cat(" - max_feats_to_change: ", private$max_feats_to_change, "\n")
      cat(" - n_synth: ", private$n_synth, "\n")
      cat(" - n_iterations: ", private$n_iterations, "\n")
      cat(" - feature_selector: ", private$feature_selector, "\n")
      cat(" - importance_method: ", private$importance_method, "\n")
      cat(" - node_selector: ", private$node_selector, "\n")
      cat(" - weight_node_selector: ", private$weight_node_selector, "\n")
    }
  )
)
