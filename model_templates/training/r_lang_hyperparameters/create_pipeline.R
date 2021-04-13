create_pipeline<-function(X, y, parameters, model_type='regression') {
  if (is.null(parameters$method)) {
    stop("method parameter value not found")
  }

  # set up dataframe for modeling
  train_df <- X
  train_df$target <- unlist(y)
  if (model_type == 'classification'){
    train_df$target <- as.factor(train_df$target)
  }

  # set up the modeling pipeline
  model_recipe <- recipe(target ~ ., data = train_df) %>%
    # Drop constant columns
    step_zv(all_predictors()) %>%
    # Numeric preprocessing
    step_medianimpute(all_numeric()) %>%
    step_normalize(all_numeric(), -all_outcomes()) %>%
    # Categorical preprocessing
    step_other(all_nominal(), -all_outcomes()) %>%
    step_dummy(all_nominal(), -all_outcomes())

  # Run the model using caret
  model <- train(model_recipe, train_df, method = parameters$method, trControl = trainControl(method = "cv", number = 3))

  return(model)
}