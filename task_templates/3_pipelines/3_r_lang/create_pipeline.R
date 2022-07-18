# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# 
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
create_pipeline<-function(X, y, model_type='regression') {
  # set up dataframe for modeling
  train_df <- X
  train_df$target <- unlist(y)
  if (model_type == 'classification'){
    train_df$target <- as.vector(as.factor(train_df$target))
  }

  # set up the modeling pipeline
  model_recipe <- recipe(target ~ ., data = train_df) %>%
    # Drop constant columns
    step_zv(all_predictors()) %>%
    # Numeric preprocessing
    step_normalize(all_numeric(), -all_outcomes()) %>%
    # Categorical preprocessing
    step_other(all_nominal(), -all_outcomes()) %>%
    step_dummy(all_nominal(), -all_outcomes())

    gbmGrid <- expand.grid(
      n.trees = 50,
      interaction.depth=1,
      shrinkage = 0.1,
      n.minobsinnode = 10
    )

    # Run the model using caret
    set.seed(1234)
    model <- train(
      model_recipe,
      train_df,
      method = "gbm",
      trControl = trainControl(method = "cv", number = 3, seeds=list(c(2,2,2), c(2,2,2), c(2,2,2), c(2))),
      tuneGrid = gbmGrid
    )

    return(model)
}
