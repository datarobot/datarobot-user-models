# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# 
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
library(stringi)

fit <- function(X, y, output_dir, class_order=NULL, row_weights=NULL, ...){}

transform <- function(X, transformer, y=NULL, ...){
    # ignore the model and output strings for each dataframe cell
    X[] <- lapply(X, function(z) { stri_rand_strings(1, 5) })
    X
}