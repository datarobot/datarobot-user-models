# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
#
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
init <- function(code_dir) {
  # custom init function to load required libraries
  library(tidyverse)
  library(caret)
  library(recipes)
  library(e1071)
  library(gbm)
}

create_pipeline<-function(X, y, model_type='regression') {
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
    step_impute_median(all_numeric()) %>%
    step_normalize(all_numeric(), -all_outcomes()) %>%
    # Categorical preprocessing
    step_other(all_nominal(), -all_outcomes()) %>%
    step_dummy(all_nominal(), -all_outcomes())

  # Run the model using caret
  set.seed(1234)
  model <- train(
    model_recipe,
    train_df,
    method = "gbm",
    trControl = trainControl(method = "cv", number = 3, seeds=list(c(2,2,2), c(2,2,2), c(2,2,2), c(2)))
  )

  return(model)
}

fit <- function(X, y, output_dir, class_order=NULL, row_weights=NULL, ...){
    #' User-provided fit method, required for custom training
    #'
    #' Trains a regression or classification model using gbm (via caret)
    #' @param X data.frame - training data to perform fit on
    #' @param y data.frame column or array - target data to perform fit on
    #' @param output_dir the path to write output. This is the path provided in '--output' parameter of the
    #' 'drum fit' command.
    #' @param class_order : A two element long list dictating the order of classes which should be used for
    #'  modeling. Class order will always be passed to fit by DataRobot for classification tasks,
    #'  and never otherwise. When models predict, they output a likelihood of one class, with a
    #'  value from 0 to 1. The likelihood of the other class is 1 - this likelihood. Class order
    #'  dictates that the first element in the list will be the 0 class, and the second will be the
    #'  1 class.
    #' @param row_weights: An array of non-negative numeric values which can be used to dictate how important
    #'  a row is. Row weights is only optionally used, and there will be no filtering for which
    #'  custom models support this. There are two situations when values will be passed into
    #'  row_weights, during smart downsampling and when weights are explicitly provided by the user
    #' @param ...: Added for forwards compatibility
    #' @return Nothing

  if(is.null(class_order)) {
    stop("class_order is required")
  }
  model <- create_pipeline(X, y, 'classification')
  # Save model
  model_path <- file.path(output_dir, 'artifact.rds')
  saveRDS(model, file = model_path)
}
