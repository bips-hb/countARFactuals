# $find_counterfactuals returns meaningful error if x_interest does not contain all columns of predictor$data$X

    Assertion on 'names(x_interest)' failed: Names must include the elements {'Sepal.Length','Sepal.Width','Petal.Length','Petal.Width'}, but is missing elements {'Sepal.Length','Sepal.Width','Petal.Length','Petal.Width'}.

# $find_counterfactuals returns meaningful error if x_interest has unexpected column types

    Columns that appear in `x_interest` and `predictor$data$X` must have the same types.

# $find_counterfactuals returns meaningful error if x_interest already has desired properties

    `x_interested` is already predicted with `desired_prob` for `desired_class`.

