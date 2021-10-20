# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# 
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
# This custom estimator task implements a GLM regressor

init <- function(code_dir) {
  # load required libraries
  library(statmod)
}

fit <- function(X, y, output_dir, class_order=NULL, row_weights=NULL, ...){
    " This hook defines how DataRobot will train this task.
    DataRobot runs this hook when the task is being trained inside a blueprint.
    As an output, this hook is expected to create an artifact containg a trained object, that is then used to score new data.
    The input parameters are passed by DataRobot based on project and blueprint configuration.

    Parameters
    -------
    X: data.frame
        Training data that DataRobot passes when this task is being trained.
    y: vector
        Project's target column.
    output_dir: str
        A path to the output folder; the artifact [in this example - containing the trained GLM] must be saved into this folder.
    class_order: list
        [Only passed for a binary estimator] a list containing names of classes: first, the one that is considered negative inside DR, then the one that is considered positive.
    row_weights: list
        A list of weights. DataRobot passes it in case of smart downsampling or when weights column is specified in project settings.

    Returns
    -------
    None
        fit() doesn't return anything, but must output an artifact (typically containing a trained object) into output_dir
        so that the trained object can be used during scoring.
    "
	  # prep data
	  train_df <- X
    train_df$target <- unlist(y)

    # train model
    e <- glm(target ~ ., data = train_df, family = gaussian())

    # Save model
    model_path <- file.path(output_dir, 'artifact.rds')
    saveRDS(strip_glm(e), file = model_path)
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
        In this example, contains trained GLM extracted from artifact.rds.
    
    Returns
    -------
    data.frame
        Returns a dataframe with scored data
        In case of regression, score() must return a dataframe with a single column with column name 'Predictions'.
    "
    return(data.frame(Predictions = predict(model, newdata=data, type = "response")))
}

strip_glm = function(cm) {
  " A helper method that removes unused data from a trained GLM making it lightweight and faster. "
  cm$y = c()
  cm$model = c()
  
  cm$residuals = c()
  cm$fitted.values = c()
  cm$effects = c()
  cm$qr$qr = c()  
  cm$linear.predictors = c()
  cm$weights = c()
  cm$prior.weights = c()
  cm$data = c()
  
  
  cm$family$variance = c()
  cm$family$dev.resids = c()
  cm$family$aic = c()
  cm$family$validmu = c()
  cm$family$simulate = c()
  attr(cm$terms,".Environment") = c()
  attr(cm$formula,".Environment") = c()
  
  return(cm)
}