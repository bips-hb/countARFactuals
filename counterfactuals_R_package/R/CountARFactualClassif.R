
CountARFactualClassif = R6::R6Class("CountARFactualClassif", 
  inherit = CounterfactualMethodClassif,
  
  public = list(
    initialize = function(predictor, max_feats_to_change = 1L, n_synth = 10L, feature_selector = "random_importance", importance_method = "fastshap") { 
      # TODO: add other hyperparameter
      super$initialize(predictor)
      checkmate::assert_integerish(max_feats_to_change, lower = 1L, upper = predictor$data$n.features)
      checkmate::assert_integerish(n_synth, lower = 1L)
      checkmate::assert_choice(feature_selector, choices = c("importance", "random", "random_importance"))
      checkmate::assert_choice(importance_method, choices = c("fastshap", "icesd"))
      private$max_feats_to_change = max_feats_to_change
      private$feature_selector = feature_selector
      private$max_feats_to_change = max_feats_to_change
      private$n_synth = n_synth
      private$importance_method = importance_method
    }
  ),
  
  private = list(
    max_feats_to_change = NULL,
    n_synth = NULL,
    feature_selector = NULL,
    importance_method = NULL,
    run = function() {
      
      # Fit ARF
      dat = copy(private$predictor$data$get.x())
      dat[, yhat := private$predictor$predict(dat)[,private$desired_class]]
      arf = adversarial_rf(dat, always.split.variables = "yhat")
      psi = forde(arf, dat)
      
      # Conditional sampling
      ##  Select conditioning set
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
            pred_wrapper = pfun, newdata = private$x_interest,
            nsim = 1000)
          vim = abs(shap[1, ])
          # ICE curve standard deviation as local importance
        } else if (private$importance_method == "icesd") {
          param_set = make_param_set(private$predictor$data$get.x())
          vim = get_ICE_sd(private$x_interest, private$predictor, param_set)
        }
      }
      
      x_interest = private$x_interest
      # TODO: which desired_prob to use when interval??
      x_interest[, yhat := mean(private$desired_prob)] 
      synth = data.table()
      for (i in 1:10) {
        feats_not_to_change = sample(seq(
          private$predictor$data$n.features - private$max_feats_to_change, 
          private$predictor$data$n.features - 1L), size = 1L)
        
        if (private$feature_selector == "importance") {
          ordered_features = names(sort(vim)) # Smallest (= less important) first
          cols = ordered_features[1:feats_not_to_change]
        } else if (private$feature_selector == "random_importance") {
          #bassert_true(all(names(vim) == private$predictor$data$feature.names))
          # get probabilities
          p_min = 0.01
          p_max = 0.99
          p_selected = 1 - ((vim - min(vim)) * (p_max - p_min) / 
              (max(vim) - min(vim) + sqrt(.Machine$double.eps)) + p_min)
          cols = sample(private$predictor$data$feature.names, 
            size = feats_not_to_change,
            prob = p_selected)
        } else if (private$feature_selector == "random") {
          cols = sample(private$predictor$data$feature.names, 
            size = feats_not_to_change)
        }
        cols = c("yhat", cols)
        # TODO: Better to condition on yhat >= target_prob (already possible)
        fixed = x_interest[, ..cols]
        synth = rbind(synth, forge(psi, n_synth = private$n_synth, evidence = fixed))
      }
      x_interest[, yhat:= NULL]
      synth[, yhat := NULL]
      
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
    },
    
    print_parameters = function() {
      #TODO
    }
  )
)
