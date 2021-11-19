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

transform <- function(data, model) {
    subset(data, select=-c(`ISO_fr`))
}

fit <- function(X, y, output_dir, class_order=NULL, row_weights=NULL, ...){
    expected_colnames <- c(
        'TPA(ADJ)_fr',
        'AVG_fr',
        'OBP_fr',
        'SLG_fr',
        'OPS_fr',
        'HRPA_fr',
        'BBPA_fr',
        'SOPA_fr',
        'KW_fr',
        'XBH/AB_fr',
        'TPA(ADJ)_so',
        'AVG_so',
        'OBP_so',
        'SLG_so',
        'OPS_so',
        'ISO_so',
        'HRPA_so',
        'BBPA_so',
        'SOPA_so',
        'KW_so',
        'XBH/AB_so',
        'TPA(ADJ)_jr',
        'AVG_jr',
        'OBP_jr',
        'SLG_jr',
        'OPS_jr',
        'ISO_jr',
        'HRPA_jr',
        'BBPA_jr',
        'SOPA_jr',
        'KW_jr',
        'XBH/AB_jr')
    if("some-weights" %in% colnames(X)) {
        expected_colnames <- c(expected_colnames, "some-weights")
    }
    stopifnot(colnames(X) == expected_colnames)
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
    return(data.frame(Predictions = predict(model, newdata=data, type = "response")))
}

strip_glm = function(cm) {
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
