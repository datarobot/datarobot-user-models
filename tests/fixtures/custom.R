# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# 
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
init <- function(...) {
    library(brnn)
    library(glmnet)
    library(dplyr)
}

transform <- function(data, model) {
    data[is.na(data)] <- 0
    if ("class" %in% names(data)) {
        data <- select(data, -class)
    }
    data
}