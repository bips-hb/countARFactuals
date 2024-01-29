
CountARFactualClassif = R6::R6Class("CountARFactualClassif", 
  inherit = CounterfactualMethodClassif,
  
  public = list(
    initialize = function(predictor, n_synth = 10L, importance_method = "fastshap") { 
      # TODO: add other hyperparameter
      super$initialize(predictor)
      checkmate::assert_integerish(n_synth, lower = 1L)
      checkmate::assert_choice(importance_method, choices = c("fastshap", "icesd"))
      private$n_synth = n_synth
      private$importance_method = importance_method
    }
  ),
  
  private = list(
    n_synth = NULL,
    importance_method = NULL,
    run = function() {
      
      # Shapley values as local importance
      pfun = function(object, newdata) { 
        unname(object$predict(newdata)[,private$desired_class])
      }
      if (private$importance_method == "fastshap") {
        if (!requireNamespace("fastshap", quietly = TRUE)) {
          stop("Package 'fastshap' needed for this measuring importance. Please install it.", call. = FALSE)
        }
        shap = explain(private$predictor, X = private$predictor$data$get.x(), 
          pred_wrapper = pfun, newdata = private$x_interest,
          nsim = 1000)
        vim = abs(shap[1, ])
      } else if (private$importance_method == "icesd") {
        param_set = counterfactuals:::make_param_set(private$predictor$data$get.x())
        vim = counterfactuals:::get_ICE_sd(private$x_interest, private$predictor, param_set)
      }
      
      ordered_features = names(sort(vim)) # Smallest (= less important) first
      
      # Fit ARF
      dat = copy(private$predictor$data$get.x())
      dat[, yhat := private$predictor$predict(dat)[,private$desired_class]]
      arf = adversarial_rf(dat, always.split.variables = "yhat")
      psi = forde(arf, dat)
      
      # Conditional sampling
      x_interest = private$x_interest
      # TODO: which desired_prob to use when interval??
      x_interest[, yhat := mean(private$desired_prob)] 
      cols = c("yhat", ordered_features[1:(length(ordered_features)-feats_to_change)])
      # TODO: Better to condition on yhat >= target_prob (already possible)
      fixed = x_interest[, ..cols]
      synth = forge(psi, n_synth = private$n_synth, evidence = fixed)
      x_interest[, yhat := NULL]

      # Recode factors to original factor levels
      factor_cols = names(which(sapply(predictor$data$X, is.factor)))
      for (factor_col in factor_cols) {
        fact_col_pred = predictor$data$X[[factor_col]]
        value =  factor(synth[[factor_col]], levels = levels(fact_col_pred), 
          ordered = is.ordered(fact_col_pred))
        set(synth, j = factor_col, value = value)
      }
      
      # Keep only valid counterfactuals
      cfs = synth[between(private$predictor$predict(synth)[,private$desired_class], 
        private$desired_prob[1L], private$desired_prob[2L]),]
      cfs[, yhat := NULL]
    },
    
    print_parameters = function() {}
  )
)
