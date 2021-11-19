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

set.seed(1)

str_right <- function(string, n) {
  substr(string, nchar(string) - (n - 1), nchar(string))
}

get_nested_path <- function(path, file_pattern){
  if(str_right(path, 1) == '/'){
    path = subbstr(path, 1, nchar(path)-1)
  }
  return(
    list.files(path=path, pattern=file_pattern, recursive = TRUE, full.names=TRUE)
  )
}

init <- function(code_dir) {
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


load_data <- function(input_filename, sparse_column_filename){
    if (grepl("\\.mtx$", input_filename)) {
        X <- readMM(input_filename)
        colnames(X) <- readLines(sparse_column_filename)
        return(X)
    }
    tmp = readChar(input_filename, file.info(input_filename)$size)
    read.csv(text=gsub("\r","", tmp, fixed=TRUE), na.strings = c("NA", ""), check.names = FALSE)
}


process_data <- function(input_filename, sparse_column_filename, target_filename, target_name,
                         num_rows, target_type){
    # read X
    df <- load_data(input_filename, sparse_column_filename)

    # run transform if present in custom model
    if (!isFALSE(transform_hook) && target_type != "transform") {
        df <- transform_hook(df, NULL)
    }

    # set num_rows
    if (num_rows == 'ALL'){
        num_rows <- nrow(df)
    } else {
        num_rows <- as.integer(num_rows)
    }

    # If a target is provided...
    X <- NULL
    y <- NULL
    na_rows <- NULL
    if (!is.null(target_filename) || !is.null(target_name)) {
        # And if targets are provided in a separate file, read them in, and treat the df as X
        if (!is.null(target_filename)) {
            X <- df
            y <- load_data(target_filename)

            # Ensure y is only a single column, and contains the same number of rows as X
            stopifnot(length(colnames(y)) == 1)
            stopifnot(nrow(df) == nrow(y))

            target_name <- colnames(y)
        # Otherwise, separate X and y from df
        } else {
            X <- df[,!(names(df) %in% c(target_name)), drop=FALSE]
            y <- df[,target_name, drop=FALSE]
        }

        # Then, drop all rows where the y value is NA
        na_rows <- as.vector(is.na(y[target_name]))
        X <- X[!na_rows,, drop=FALSE]
        y <- y[!na_rows,, drop=FALSE]
    # If no target is provided, then treat the df as X
    } else {
        X <- df
        na_rows <- rep(FALSE, nrow(X))  # Set all of na_rows to FALSE since no y values are NA
    }

    # Sample X (and y if provided) using the provided num_rows amount
    sample_rows <- as.vector(sample(nrow(X), size=num_rows))
    X <- X[sample_rows,, drop=FALSE]
    if (!is.null(y)) {
        y <- y[sample_rows,, drop=TRUE]  # drop here so y is a single dimension
    }

    return(list('X' = X, 'y' = y, 'na_rows' = na_rows, 'sample_rows' = sample_rows))

}


process_weights <- function(X, weights_filename, weights, na_rows, sample_rows){
  row_weights <- NULL
  if (!is.null(weights_filename)){
    row_weights <- load_data(weights_filename)

  } else if(!is.null(weights)){
    if (! weights %in% colnames(X)){
      stop( paste("The column name",
                  weights,
                  "is not one of the columns in your training data", sep=' '))
    }
    row_weights <- X[, weights, drop=FALSE]
  } else {
    row_weights = NULL
  }

  # Drop the NA rows found from y (if applicable) then use the same sample_rows as X (and y if present)
  if (!is.null(row_weights)) {
    row_weights <- row_weights[!na_rows,, drop=FALSE]
    row_weights <- row_weights[sample_rows,, drop=FALSE]
  }
  row_weights
}


process_parameters <- function(parameter_filename){
    if (!is.null(parameter_filename)){
        parameters = fromJSON(file=parameter_filename)
    } else {
        parameters = NULL
    }
    parameters
}

#' Fits and saves a model using user-provided Fit method
#'
#' @param data data.frame from which to train model
#' @param output_dir directory to save model
#' @param class_order array containing [negative label, positive label] or NULL
#' @param row_weights array with row weights, or NULL
#'

outer_fit <- function(output_dir, input_filename, sparse_column_filename, target_filename,
                      target_name, num_rows, weights_filename, weights,
                      positive_class_label, negative_class_label, class_labels, parameter_filename,
                      target_type) {

    processed_data <- process_data(input_filename, sparse_column_filename, target_filename,
                                   target_name, num_rows, target_type)

    X <- processed_data$X
    y <- processed_data$y
    na_rows <- processed_data$na_rows
    sample_rows <- processed_data$sample_rows

    row_weights <- process_weights(X, weights_filename, weights, na_rows, sample_rows)

    parameters <- process_parameters(parameter_filename)

    if (!is.null(positive_class_label) && !is.null(negative_class_label)){
        class_order <- c(negative_class_label, positive_class_label)
    } else if (!is.null(class_labels)) {
        class_order <- class_labels
    } else {
        class_order <- NULL
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

