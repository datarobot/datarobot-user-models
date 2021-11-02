# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
#
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
fit <- function(X, y, output_dir, class_order=NULL, row_weights=NULL, ...){}

transform <- function(X, transformer, y=NULL, ...){
    stopifnot(!is.null(colnames(X)))
    # Ignore the model and convert the sparse input to dense
    as.data.frame(as.matrix(X))
}
