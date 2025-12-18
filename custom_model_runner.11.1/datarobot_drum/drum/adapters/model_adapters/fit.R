# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
#
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
# Needed libraries
library(caret)
library(devtools)
library(Matrix)
library(rjson)

init_hook <- FALSE
fit_hook <- FALSE
transform_hook <- FALSE

TARGET_TYPE <- NULL

set.seed(1)

str_right <- function(string, n) {
    substr(string, nchar(string) - (n - 1), nchar(string))
}

get_nested_path <- function(path, file_pattern){
    if(str_right(path, 1) == '/'){
        path <- subbstr(path, 1, nchar(path)-1)
    }
    return(
        list.files(path=path, pattern=file_pattern, recursive = TRUE, full.names=TRUE)
    )
}

outer_init <- function(code_dir, target_type) {
    TARGET_TYPE <<- target_type
    custom_path <- get_nested_path(code_dir, 'custom.[Rr]')
    if(length(custom_path) >= 1){
        custom_loaded <- import(custom_path[1])
    } else{
        stop('No custom file found.')
    }
    if (isTRUE(custom_loaded)) {
        init_hook <<- getHookMethod("init")
        fit_hook <<- getHookMethod("fit")
        transform_hook <<- getHookMethod("transform")
    }

  if (!isFALSE(init_hook)) {
    init_hook(code_dir=code_dir)
  }
}


#' Fits and saves a model using user-provided Fit method
#'
#' @param X data.frame from which to train model
#' @param y vector containing target values, or NULL
#' @param output_dir directory to save model
#' @param class_order array containing [negative label, positive label] or NULL
#' @param row_weights array with row weights, or NULL
#' @param parameters list containing hyperparameters, or NULL
#'

outer_fit <- function(X, y, output_dir, class_order, row_weights, parameters) {
    if (!isFALSE(transform_hook) && TARGET_TYPE != "transform") {
        X <- transform_hook(X, NULL)
    }

    if (!isFALSE(fit_hook)) {
        kwargs <- list()
        kwargs <- append(kwargs, list(X=X,
                                      y=y,
                                      output_dir=output_dir,
                                      row_weights=row_weights,
                                      class_order=class_order,
                                      parameters=parameters))

        do.call(fit_hook, kwargs)
    } else {
        stop(sprintf("No Fit method provided."))
    }
}

