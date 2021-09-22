create_pipeline<-function(X) {
  # set up dataframe for modeling
  train_df <- X

  preProcValues <- preProcess(train_df, method = c("medianImpute", "center", "scale"))
  trainTransformed <- predict(preProcValues, train_df)
  # Run the model using caret
  model = isolationForest$new()
  model$fit(trainTransformed)
  return (model)
}