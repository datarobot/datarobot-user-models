# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# 
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
library(recipes)


transform <- function(X, transformer, ...){
  "
  This hook defines how DataRobot will use the trained object from fit() to transform new data.
  As an output, this hook is expected to return the transformed data.

  Parameters
  -------
  X: data.frame
      Data that DataRobot passes for transforming.
  transformer: Any
      Trained object, extracted by DataRobot from the artifact created in fit().
      In this example, transformer contains the median values stored in a vector.

  Returns
  -------
  data.frame
      Returns the transformed dataframe
  "
  X_median <- transformer

  # replace NA values with the median values for each column
  for (i in 1:ncol(X)) {
    X[is.na(X[,i]), i] <- X_median[i]
  }
  out <- Matrix(as.matrix(X), sparse=TRUE)
  f <- function(i) paste("feature_", i, sep="")
  colnames(out) <- sapply(0:(ncol(out)-1), f)
  out
}
