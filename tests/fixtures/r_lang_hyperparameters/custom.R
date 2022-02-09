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
  library(caret)
  library(e1071)
}

fit <- function(X, y, output_dir, parameters=NULL, ...){
    #' User-provided fit method, required for custom training
    #'
    #' Trains a regression or classification model using gbm (via caret)
    #' @param X data.frame - training data to perform fit on
    #' @param y data.frame column or array - target data to perform fit on
    #' @param output_dir the path to write output. This is the path provided in '--output' parameter of the
    #' 'drum fit' command.
    #' @param parameters: A dataframe containing parameter name to value mapping
    #' @param ...: Added for forwards compatibility
    #' @return Nothing

  if (is.null(parameters)) {
    stop("parameters not found")
  }

  svmParameters <- expand.grid(cost = parameters$cost, weight = parameters$weight)
  model <- train(X, y = make.names(y), method = "svmLinearWeights", tuneGrid = svmParameters, trControl = trainControl(classProbs = TRUE))

  # Save model
  model_path <- file.path(output_dir, 'artifact.rds')
  saveRDS(model, file = model_path)
}

score <- function(data, model, ...){
  " This hook defines how DataRobot will use the trained object from fit() to score new data.
  DataRobot runs this hook when the task is used for scoring inside a blueprint.
  As an output, this hook is expected to return the scored data.
  The input parameters are passed by DataRobot based on dataset and blueprint configuration.

  Parameters
  -------
  data: data.frame
      Data that DataRobot passes for scoring.
  model: Any
      Trained object, extracted by DataRobot from the artifact created in fit().
      In this example, contains trained SVM extracted from artifact.rds.

  Returns
  -------
  data.frame
      Returns a dataframe with scored data
      In case of classification (binary or multiclass), must return a dataframe with a column per class
      with class names used as column names
      and probabilities of classes as values (each row must sum to 1.0)
  "
  scores <- predict(model, newdata = data, type = "prob")
  names(scores) <- c('0', '1')
  return(scores)
}
